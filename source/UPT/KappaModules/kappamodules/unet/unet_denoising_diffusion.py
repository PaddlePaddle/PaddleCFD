from functools import partial

import paddle

from kappamodules.attention import (DotProductAttention1d,
                                    DotProductAttention2d,
                                    DotProductAttention3d,
                                    EfficientAttention1d, EfficientAttention2d,
                                    EfficientAttention3d)
from kappamodules.layers import Identity, RMSNorm
from kappamodules.modulation import Film


class ResnetBlock(paddle.nn.Layer):
    def __init__(self, dim_in, dim_out, conv_ctor, dim_cond=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_cond = dim_cond
        if dim_cond is not None:
            self.film = Film(dim_cond=dim_cond, dim_out=dim_out)
        else:
            self.film = Identity()
        self.conv1 = conv_ctor(dim_in, dim_out, kernel_size=3, padding=1)
        self.norm1 = paddle.nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.conv2 = conv_ctor(dim_out, dim_out, kernel_size=3, padding=1)
        self.norm2 = paddle.nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.residual = (
            conv_ctor(dim_in, dim_out, 1) if dim_in != dim_out else paddle.nn.Identity()
        )

    def forward(self, x, cond=None):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film(x, cond=cond)
        x = paddle.nn.functional.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual


class UpsampleConv(paddle.nn.Layer):
    def __init__(
        self, dim_in, dim_out, conv_ctor, kernel_size=3, scale_factor=2, mode="nearest"
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.upsample = paddle.nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv = conv_ctor(
            dim_in, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


def pre_and_post_norm_attention(dim, attn_ctor):
    return paddle.nn.Sequential(RMSNorm(dim), attn_ctor(dim), RMSNorm(dim))


def post_norm_attention(dim, attn_ctor):
    return paddle.nn.Sequential(attn_ctor(dim), RMSNorm(dim))


class UnetDenoisingDiffusion(paddle.nn.Layer):
    """
    reimplementation of https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    each block downsamples the input by a factor of 2 while doubling the dimension
    first up and last down block dont scale dimension
    each block consists of 2 residual blocks (each with 2 conv layers) and a linear attention block
    the last down block and the first up block have dot product attention
    in the middle there are 2 residual blocks (without up/downsampling) with a dot product attention in between
    if dim_cond is not None -> add FILM modulation to all residual blocks
    stem was modified to not downsample the input -> the input is expected to be a latent representation
    """

    def __init__(
        self, dim, dim_in, ndim, depth, num_heads=None, dim_out=None, dim_cond=None
    ):
        super().__init__()
        self.dim = dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ndim = ndim
        self.num_heads = num_heads
        self.depth = depth
        if ndim == 1:
            conv_ctor = paddle.nn.Conv1d
            linear_attn_ctor = EfficientAttention1d
            dot_product_attn_ctor = DotProductAttention1d
        elif ndim == 2:
            conv_ctor = paddle.nn.Conv2d
            linear_attn_ctor = EfficientAttention2d
            dot_product_attn_ctor = DotProductAttention2d
        elif ndim == 3:
            conv_ctor = paddle.nn.Conv3d
            linear_attn_ctor = EfficientAttention3d
            dot_product_attn_ctor = DotProductAttention3d
        else:
            raise NotImplementedError
        if num_heads is None:
            linear_attn_ctor = Identity
            dot_product_attn_ctor = Identity
        block_ctor = partial(ResnetBlock, conv_ctor=conv_ctor, dim_cond=dim_cond)
        upsample_conv_ctor = partial(UpsampleConv, conv_ctor=conv_ctor)
        linear_attn_ctor = partial(
            pre_and_post_norm_attention,
            attn_ctor=partial(
                linear_attn_ctor, num_heads=num_heads, channel_first=True
            ),
        )
        dot_product_attn_ctor = partial(
            post_norm_attention,
            attn_ctor=partial(
                dot_product_attn_ctor, num_heads=num_heads, channel_first=True
            ),
        )
        self.stem = conv_ctor(dim_in, dim, kernel_size=3, padding=1)
        in_dims = []
        out_dims = []
        attn_ctors = []
        strides = []
        for i in range(depth):
            if i == 0:
                in_dims.append(dim)
                out_dims.append(dim)
            else:
                in_dim = dim * 2 ** (i - 1)
                in_dims.append(in_dim)
                out_dims.append(in_dim * 2)
            if i < depth - 1:
                attn_ctors.append(linear_attn_ctor)
            else:
                attn_ctors.append(dot_product_attn_ctor)
            if i < depth - 1:
                strides.append(2)
            else:
                strides.append(1)
        self.down_blocks = paddle.nn.ModuleList()
        for i in range(depth):
            dim_in = in_dims[i]
            dim_out = out_dims[i]
            attn_ctor = attn_ctors[i]
            stride = strides[i]
            if stride == 1:
                downsample_ctor = partial(conv_ctor, kernel_size=3, padding=1)
            else:
                assert stride == 2
                downsample_ctor = partial(conv_ctor, kernel_size=2, stride=2)
            block = paddle.nn.ModuleList(
                [
                    block_ctor(dim_in=dim_in, dim_out=dim_in),
                    block_ctor(dim_in=dim_in, dim_out=dim_in),
                    attn_ctor(dim_in),
                    downsample_ctor(dim_in, dim_out),
                ]
            )
            self.down_blocks.append(block)
        mid_dim = out_dims[-1]
        self.mid_block1 = block_ctor(dim_in=mid_dim, dim_out=mid_dim)
        self.mid_attn = dot_product_attn_ctor(mid_dim)
        self.mid_block2 = block_ctor(dim_in=mid_dim, dim_out=mid_dim)
        self.up_blocks = paddle.nn.ModuleList()
        for i in reversed(range(depth)):
            dim_in = in_dims[i]
            dim_out = out_dims[i]
            attn_ctor = attn_ctors[i]
            stride = strides[depth - 1 - i]
            if stride == 1:
                upsample_ctor = partial(conv_ctor, kernel_size=3, padding=1)
            else:
                assert stride == 2
                upsample_ctor = upsample_conv_ctor
            block = paddle.nn.ModuleList(
                [
                    block_ctor(dim_in=dim_in + dim_out, dim_out=dim_out),
                    block_ctor(dim_in=dim_in + dim_out, dim_out=dim_out),
                    attn_ctor(dim_out),
                    upsample_ctor(dim_out, dim_in),
                ]
            )
            self.up_blocks.append(block)
        self.final_res_block = block_ctor(dim_in=dim * 2, dim_out=dim)
        if self.dim_out is None:
            self.final_conv = paddle.nn.Identity()
        else:
            self.final_conv = conv_ctor(dim, self.dim_out, kernel_size=1)

    def forward(self, x, cond=None):
        x = self.stem(x)
        stack = [x]
        for block1, block2, attn, downsample in self.down_blocks:
            x = block1(x, cond=cond)
            stack.append(x)
            x = block2(x, cond=cond)
            x = attn(x) + x
            stack.append(x)
            x = downsample(x)
        x = self.mid_block1(x, cond=cond)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, cond=cond)
        for block1, block2, attn, upsample in self.up_blocks:
            x = paddle.cat((x, stack.pop()), dim=1)
            x = block1(x, cond=cond)
            x = paddle.cat((x, stack.pop()), dim=1)
            x = block2(x, cond=cond)
            x = attn(x) + x
            x = upsample(x)
        x = paddle.cat((x, stack.pop()), dim=1)
        x = self.final_res_block(x, cond=cond)
        x = self.final_conv(x)
        return x
