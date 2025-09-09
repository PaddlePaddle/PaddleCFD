import math
from functools import partial

import einops
import numpy as np
import paddle
from einops import rearrange
from einops.layers.paddle import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


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


class Residual(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


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


class WeightStandardizedConv2d(paddle.nn.Conv2D):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-05 if x.dtype == "float32" else 0.001
        weight = self.weight
        mean = einops.reduce(weight, "o ... -> o 1 1 1", "mean")
        var = einops.reduce(weight, "o ... -> o 1 1 1", partial(paddle.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return paddle.nn.functional.conv2d(
            x=x,
            weight=normalized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LayerNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.g = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, x):
        eps = 1e-05 if x.dtype == paddle.float32 else 0.001
        var = paddle.var(x=x, axis=1, unbiased=False, keepdim=True)
        mean = paddle.mean(x=x, axis=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(paddle.nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)
        # return self.fn(x)
        return x


class SinusoidalPosEmb(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(paddle.nn.Layer):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=half_dim), trainable=not is_random
        )

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = paddle.concat(x=(freqs.sin(), freqs.cos()), axis=-1)
        fouriered = paddle.concat(x=(x, fouriered), axis=-1)
        return fouriered


class Block(paddle.nn.Layer):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        self.norm = paddle.nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        # self.norm = LayerNorm(dim_out)
        self.act = paddle.nn.Silu()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(paddle.nn.Layer):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            paddle.nn.Sequential(
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
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
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head**-0.5
            hidden_dim = dim_head * heads
            self.to_qkv = paddle.nn.Conv2D(
                in_channels=dim,
                out_channels=hidden_dim * 3,
                kernel_size=1,
                bias_attr=False,
            )
            self.to_out = paddle.nn.Sequential(
                paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
                LayerNorm(dim),
            )

    def forward(self, x):
        if self.heads is None:
            return x
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q * self.scale
        v = v / (h * w)
        k = k.astype(paddle.float16)
        v = v.astype(paddle.float16)
        context = paddle.einsum("b h d n, b h e n -> b h d e", k, v).astype(paddle.float32)
        out = paddle.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(paddle.nn.Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head**-0.5
            hidden_dim = dim_head * heads
            self.to_qkv = paddle.nn.Conv2D(
                in_channels=dim,
                out_channels=hidden_dim * 3,
                kernel_size=1,
                bias_attr=False,
            )
            self.to_out = paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        if self.heads is None:
            return x
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q * self.scale
        q = q.astype(paddle.float16)
        k = k.astype(paddle.float16)
        sim = paddle.einsum("b h d i, b h d j -> b h i j", q, k).astype(paddle.float32)
        attn = sim
        out = paddle.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = self.to_out(out)
        return out


class Unet2D(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        Par,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attention_heads=4,
        dim_head=32,
    ):
        super().__init__()
        self.Par = Par
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
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = paddle.nn.Sequential(
            sinu_pos_emb,
            paddle.nn.Linear(in_features=fourier_dim, out_features=time_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=time_dim, out_features=time_dim),
        )

        kwargs_block_klass = {"time_emb_dim": time_dim, "groups": resnet_block_groups}
        self.downs = paddle.nn.LayerList(sublayers=[])
        self.ups = paddle.nn.LayerList(sublayers=[])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(
                paddle.nn.LayerList(
                    [
                        ResnetBlock(dim_in, dim_in, **kwargs_block_klass),
                        ResnetBlock(dim_in, dim_in, **kwargs_block_klass),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(dim_in, heads=attention_heads, dim_head=dim_head),
                            )
                        ),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else paddle.nn.Conv2D(
                                dim_in,
                                dim_out,
                                kernel_size=3,
                                padding=1,
                            )
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, **kwargs_block_klass)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=attention_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, **kwargs_block_klass)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(
                paddle.nn.LayerList(
                    [
                        ResnetBlock(dim_out + dim_in, dim_out, **kwargs_block_klass),
                        ResnetBlock(dim_out + dim_in, dim_out, **kwargs_block_klass),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(dim_out, heads=attention_heads, dim_head=dim_head),
                            )
                        ),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else paddle.nn.Conv2D(
                                dim_out,
                                dim_in,
                                kernel_size=3,
                                padding=1,
                            )
                        ),
                    ]
                )
            )
        self.out_dim = self.Par["nf"]
        self.final_res_block = ResnetBlock(dim * 2, dim, **kwargs_block_klass)
        self.final_conv = paddle.nn.Conv2D(in_channels=dim, out_channels=self.out_dim, kernel_size=1)

    def get_grid(self, shape, device="cuda"):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = paddle.to_tensor(data=np.linspace(0, 1, size_x), dtype="float32")
        gridx = gridx.reshape(1, size_x, 1, 1).tile(repeat_times=[batchsize, 1, size_y, 1])
        gridy = paddle.to_tensor(data=np.linspace(0, 1, size_y), dtype="float32")
        gridy = gridy.reshape(1, 1, size_y, 1).tile(repeat_times=[batchsize, size_x, 1, 1])
        return paddle.concat(x=(gridx, gridy), axis=-1).to(device)

    def forward(self, x, time, x_self_cond=None, use_grid=True):
        x = (x - self.Par["inp_shift"]) / self.Par["inp_scale"]
        x = x.reshape([-1, self.Par["lb"] * self.Par["nf"], self.Par["nx"], self.Par["ny"]])

        time = (time - self.Par["t_shift"]) / self.Par["t_scale"]

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time) if time is not None else None
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = paddle.concat(x=(x, h.pop()), axis=1)
            x = block1(x, t)
            x = paddle.concat(x=(x, h.pop()), axis=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = paddle.concat(x=(x, r), axis=1)
        x = self.final_res_block(x, t)

        out = self.final_conv(x)
        out = out.unsqueeze(axis=1)
        out = out * self.Par["out_scale"] + self.Par["out_shift"]
        out = out.reshape([-1, self.Par["nf"], self.Par["nx"], self.Par["ny"]]) * self.Par["mask"]
        return out


if __name__ == "__main__":
    model = Unet2D(dim=16, dim_mults=(1, 2, 4, 8))
    pred = model(paddle.rand(shape=(16, 3, 64, 64)), time=paddle.rand(shape=(16,)))
    print("OK")
