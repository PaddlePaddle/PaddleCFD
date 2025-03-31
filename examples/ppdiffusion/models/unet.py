from functools import partial
from typing import Optional

import paddle
from einops import rearrange, reduce

from .base_model import BaseModel
from .model_utils import default
from .modules.attention import Attention, LinearAttention
from .modules.misc import Residual, get_time_embedder
from .modules.net_norm import PreNorm


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
    return paddle.nn.Conv2D(
        in_channels=dim,
        out_channels=default(dim_out, dim),
        kernel_size=4,
        stride=2,
        padding=1,
    )


class WeightStandardizedConv2d(paddle.nn.Conv2D):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-05 if x.dtype == "float32" else 0.001
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(paddle.var, unbiased=False))
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
        self.g = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=[1, dim, 1, 1]))

    def forward(self, x):
        eps = 1e-05 if x.dtype == "float32" else 0.001
        var = paddle.var(x=x, axis=1, unbiased=False, keepdim=True)
        mean = paddle.mean(x=x, axis=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Block(paddle.nn.Layer):
    def __init__(self, dim, dim_out, groups=8, dropout: float = 0.0):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = paddle.nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.act = paddle.nn.Silu()
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        double_conv_layer: bool = True,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
    ):
        super().__init__()
        self.mlp = (
            paddle.nn.Sequential(
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2),
            )
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups, dropout=dropout1)
        self.block2 = (
            Block(dim_out, dim_out, groups=groups, dropout=dropout2) if double_conv_layer else paddle.nn.Identity()
        )
        self.residual_conv = (
            paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=1)
            if dim != dim_out
            else paddle.nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(chunks=2, axis=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.residual_conv(x)


class Unet(BaseModel):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4, 8),
        num_conditions: int = 0,
        resnet_block_groups=8,
        with_time_emb: bool = False,
        block_dropout: float = 0.0,  # for second block in resnet block
        block_dropout1: float = 0.0,  # for first block in resnet block
        attn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        double_conv_layer: bool = True,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 128)
        keep_spatial_dims: bool = False,
        init_kernel_size: int = 7,
        init_padding: int = 3,
        init_stride: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # determine dimensions
        input_channels = self.num_input_channels + self.num_conditional_channels
        output_channels = self.num_output_channels or input_channels
        self.save_hyperparameters()
        if num_conditions >= 1:
            assert (
                self.num_conditional_channels > 0
            ), f"num_conditions is {num_conditions} but num_conditional_channels is {self.num_conditional_channels}"
        init_dim = default(init_dim, dim)
        assert (
            upsample_dims is None
            and outer_sample_mode is None
            or upsample_dims is not None
            and outer_sample_mode is not None
        ), "upsample_dims and outer_sample_mode must be both None or both not None"
        if outer_sample_mode is not None:
            self.upsampler = paddle.nn.Upsample(size=tuple(upsample_dims), mode=self.outer_sample_mode)
        else:
            self.upsampler = None
        self.init_conv = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=init_dim,
            kernel_size=init_kernel_size,
            padding=init_padding,
            stride=init_stride,
        )
        self.dropout_input = paddle.nn.Dropout(p=input_dropout)
        self.dropout_input_for_residual = paddle.nn.Dropout(p=input_dropout)
        if with_time_emb:
            self.time_dim = dim * 2
            self.time_emb_mlp = get_time_embedder(self.time_dim, dim, learned_sinusoidal_cond, learned_sinusoidal_dim)
        else:
            self.time_dim = None
            self.time_emb_mlp = None
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            dropout2=block_dropout,
            dropout1=block_dropout1,
            double_conv_layer=double_conv_layer,
            time_emb_dim=self.time_dim,
        )
        # layers
        self.downs = paddle.nn.LayerList(sublayers=[])
        self.ups = paddle.nn.LayerList(sublayers=[])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            do_downsample = not is_last and not keep_spatial_dims
            sublayers = [
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Residual(
                    PreNorm(
                        dim_in,
                        fn=LinearAttention(dim_in, rescale="qkv", dropout=attn_dropout),
                        norm=LayerNorm,
                    )
                ),
            ]
            if do_downsample:
                sublayers.append(Downsample(dim_in, dim_out))
            else:
                sublayers.append(
                    paddle.nn.Conv2D(
                        in_channels=dim_in,
                        out_channels=dim_out,
                        kernel_size=3,
                        padding=1,
                    )
                )
            self.downs.append(paddle.nn.LayerList(sublayers))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, fn=Attention(mid_dim, dropout=attn_dropout), norm=LayerNorm))
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            do_upsample = not is_last and not keep_spatial_dims
            sublayers = [
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Residual(
                    PreNorm(
                        dim_out,
                        fn=LinearAttention(dim_out, rescale="qkv", dropout=attn_dropout),
                        norm=LayerNorm,
                    )
                ),
            ]
            if do_upsample:
                sublayers.append(Upsample(dim_out, dim_in))
            else:
                sublayers.append(
                    paddle.nn.Conv2D(
                        in_channels=dim_out,
                        out_channels=dim_in,
                        kernel_size=3,
                        padding=1,
                    )
                )
            self.ups.append(paddle.nn.LayerList(sublayers))

        default_out_dim = input_channels * (1 if not learned_variance else 2)
        self.out_dim = default(output_channels, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = self.get_head()
        if hasattr(self, "spatial_shape") and self.spatial_shape is not None:
            b, s1, s2 = 1, *self.spatial_shape
            self.example_input_array = [
                paddle.rand(shape=[b, self.num_input_channels, s1, s2]),
                paddle.rand(shape=b) if with_time_emb else None,
                (
                    paddle.rand(shape=[b, self.num_conditional_channels, s1, s2])
                    if self.num_conditional_channels > 0
                    else None
                ),
            ]

    def get_head(self):
        return paddle.nn.Conv2D(in_channels=self.hparams.dim, out_channels=self.out_dim, kernel_size=1)

    def set_head_to_identity(self):
        self.final_conv = paddle.nn.Identity()

    def get_block(self, dim_in, dim_out, dropout: Optional[float] = None):
        return ResnetBlock(
            dim_in,
            dim_out,
            groups=self.hparams.resnet_block_groups,
            dropout1=dropout or self.hparams.block_dropout1,
            dropout2=dropout or self.hparams.block_dropout,
            time_emb_dim=self.time_dim,
        )

    def get_extra_last_block(self, dropout: Optional[float] = None):
        return self.get_block(self.hparams.dim, self.hparams.dim, dropout=dropout)

    def forward(self, x, time=None, condition=None, return_time_emb: bool = False):
        if self.num_conditional_channels > 0:
            # condition = default(condition, lambda: torch.zeros_like(x))
            x = paddle.concat(x=(condition, x), axis=1)
        else:
            assert condition is None, "condition is not None but num_conditional_channels is 0"
        orig_x_shape = tuple(x.shape)[-2:]
        x = self.upsampler(x) if self.upsampler is not None else x
        x = self.init_conv(x)
        r = self.dropout_input_for_residual(x) if self.hparams.input_dropout > 0 else x.clone()
        x = self.dropout_input(x)
        t = self.time_emb_mlp(time) if self.time_emb_mlp is not None else None
        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
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
        x = self.final_conv(x)
        if self.upsampler is not None:
            # x = F.interpolate(x, orig_x_shape, mode='bilinear', align_corners=False)
            x = paddle.nn.functional.interpolate(x=x, size=orig_x_shape, mode=self.hparams.outer_sample_mode)
        if return_time_emb:
            return x, t
        return x
