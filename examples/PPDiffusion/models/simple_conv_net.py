from typing import Sequence

import paddle
from einops import rearrange

from .base_model import BaseModel
from .model_utils import get_normalization_layer
from .modules.misc import SinusoidalPosEmb


class ConvBlock(paddle.nn.Layer):
    """A simple convolutional block with BatchNorm and GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        residual: bool = False,
        time_emb_dim: int = None,
        net_normalization: str = "batch_norm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = get_normalization_layer(net_normalization, out_channels, num_groups=32)
        self.activation = paddle.nn.GELU()  # a non-linearity
        self.residual = residual and in_channels == out_channels
        self.time_mlp = (
            paddle.nn.Sequential(
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=time_emb_dim, out_features=out_channels * 2),
            )
            if time_emb_dim is not None
            else None
        )
        self.dropout = paddle.nn.Dropout(p=dropout) if dropout > 0 else paddle.nn.Identity()

    def forward(self, x: paddle.Tensor, time_emb=None) -> paddle.Tensor:
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        if self.time_mlp is not None:
            assert time_emb is not None
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(chunks=2, axis=1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        x = self.dropout(x)
        if self.residual:
            x = x + residual
        return x


# The model:
class SimpleConvNet(BaseModel):
    """A simple convolutional network."""

    def __init__(
        self,
        dim: int,
        with_time_emb: bool = False,
        net_normalization: str = "batch_norm",
        kernel_sizes: Sequence[int] = (7, 3, 3),
        keep_spatial_shape: bool = True,
        residual=True,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        in_channels = self.num_input_channels + self.num_conditional_channels
        if with_time_emb:
            self.time_dim = dim * 2
            self.time_emb_mlp = paddle.nn.Sequential(
                SinusoidalPosEmb(dim),
                paddle.nn.Linear(in_features=dim, out_features=self.time_dim),
                paddle.nn.GELU(),
                paddle.nn.Linear(in_features=self.time_dim, out_features=self.time_dim),
            )
        else:
            self.time_dim = None
            self.time_emb_mlp = None
        convs = []
        conv_kwargs = dict(
            net_normalization=net_normalization,
            time_emb_dim=self.time_dim,
            dropout=dropout,
        )
        for lay_idx, kernel_size in enumerate(kernel_sizes):
            input_channels = in_channels if lay_idx == 0 else dim
            padding = (kernel_size - 1) // 2 if keep_spatial_shape else 0
            convs += [
                ConvBlock(
                    input_channels, dim, kernel_size=kernel_size, padding=padding, residual=residual, **conv_kwargs
                )
            ]
        self.convs = paddle.nn.LayerList(sublayers=convs)
        self.head = paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=self.num_output_channels,
            kernel_size=1,
            padding=0,
        )
        if hasattr(self, "spatial_shape") and self.spatial_shape is not None:
            b, s1, s2 = 1, self.spatial_shape[0], self.spatial_shape[1]
            self.example_input_array = [
                paddle.rand(shape=[b, self.num_input_channels, s1, s2]),
                paddle.rand(shape=b) if with_time_emb else None,
                (
                    paddle.rand(shape=[b, self.num_conditional_channels, s1, s2])
                    if self.num_conditional_channels > 0
                    else None
                ),
            ]

    def forward(self, inputs, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        """
        Args:
            inputs: a batch of images of shape (batch_size, channels_in, height, width)

        Returns:
            y: a batch of images of shape (batch_size, channels_out, height, width)
        """
        if self.num_conditional_channels > 0:
            x = paddle.concat(x=[inputs, condition], axis=1)
        else:
            x = inputs
            assert condition is None
        t = self.time_emb_mlp(time) if self.time_emb_mlp is not None else None
        for layer in self.convs:
            x = layer(x, t)
        y = self.head(x)
        return y
