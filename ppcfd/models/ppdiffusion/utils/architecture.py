import math
from collections import namedtuple
from functools import partial

import paddle
from einops import rearrange
from einops import repeat
from einops.layers.paddle import Rearrange

from .attend import Attend


__version__ = "2.0.12"
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return numer % denom == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return math.sqrt(num) ** 2 == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def Upsample(dim, dim_out=None):
    return paddle.nn.Sequential(
        paddle.nn.Upsample(scale_factor=2, mode="nearest"),
        paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=default(dim_out, dim),
            kernel_size=3,
            padding=1,
        ),
    )


def Downsample(dim, dim_out=None):
    return paddle.nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        paddle.nn.Conv2D(in_channels=dim * 4, out_channels=default(dim_out, dim), kernel_size=1),
    )


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, x):
        return paddle.nn.functional.normalize(x=x, axis=1) * self.g * self.scale


class SinusoidalPosEmb(paddle.nn.Layer):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(paddle.nn.Layer):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = self.create_parameter(shape=[half_dim], default_initializer=paddle.nn.initializer.Normal())
        self.weights.stop_gradient = is_random

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = paddle.concat(x=(freqs.sin(), freqs.cos()), axis=-1)
        fouriered = paddle.concat(x=(x, fouriered), axis=-1)
        return fouriered


class Block(paddle.nn.Layer):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = paddle.nn.Silu()
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(paddle.nn.Layer):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (
            paddle.nn.Sequential(
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = (
            paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=1)
            if dim != dim_out
            else paddle.nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(chunks=2, axis=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(paddle.nn.Layer):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.mem_kv = self.create_parameter(
            shape=[2, heads, dim_head, num_mem_kv], default_initializer=paddle.nn.initializer.Normal()
        )
        self.to_qkv = paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=hidden_dim * 3,
            kernel_size=1,
            bias_attr=False,
        )
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
            RMSNorm(dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(paddle.concat, axis=-1), ((mk, k), (mv, v)))
        q = paddle.nn.functional.softmax(q, axis=-2)
        k = paddle.nn.functional.softmax(k, axis=-1)
        q = q * self.scale
        context = paddle.einsum("b h d n, b h e n -> b h d e", k, v)
        out = paddle.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(paddle.nn.Layer):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)
        self.mem_kv = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[2, heads, num_mem_kv, dim_head])
        )
        self.to_qkv = paddle.nn.Conv2D(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias_attr=False)
        self.to_out = paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = tuple(x.shape)
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads),
            qkv,
        )
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(paddle.concat, axis=-2), ((mk, k), (mv, v)))
        out = self.attend(q, k, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,
        flash_attn=False,
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = paddle.nn.Conv2D(in_channels=input_channels, out_channels=init_dim, kernel_size=7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim
        self.time_mlp = paddle.nn.Sequential(
            sinu_pos_emb,
            paddle.nn.Linear(in_features=fourier_dim, out_features=time_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=time_dim, out_features=time_dim),
        )
        if not full_attn:
            full_attn = *((False,) * (len(dim_mults) - 1)), True
        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        assert len(full_attn) == len(dim_mults)
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)
        self.downs = paddle.nn.LayerList(sublayers=[])
        self.ups = paddle.nn.LayerList(sublayers=[])
        num_resolutions = len(in_out)
        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= num_resolutions - 1
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.downs.append(
                paddle.nn.LayerList(
                    sublayers=[
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else paddle.nn.Conv2D(
                                in_channels=dim_in,
                                out_channels=dim_out,
                                kernel_size=3,
                                padding=1,
                            )
                        ),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)
        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == len(in_out) - 1
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.ups.append(
                paddle.nn.LayerList(
                    sublayers=[
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_klass(
                            dim_out,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else paddle.nn.Conv2D(
                                in_channels=dim_out,
                                out_channels=dim_in,
                                kernel_size=3,
                                padding=1,
                            )
                        ),
                    ]
                )
            )
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = paddle.nn.Conv2D(in_channels=init_dim, out_channels=self.out_dim, kernel_size=1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all(
            divisible_by(d, self.downsample_factor) for d in x.shape[-2:]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: paddle.zeros_like(x=x))
            x = paddle.concat(x=(x_self_cond, x), axis=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = paddle.concat(x=(x, h.pop()), axis=1)
            x = block1(x, t)
            x = paddle.concat(x=(x, h.pop()), axis=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)
        x = paddle.concat(x=(x, r), axis=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
