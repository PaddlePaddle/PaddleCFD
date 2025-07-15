# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from einops import rearrange

from ppcfd.models.ppdiffusion import BaseModel
from ppcfd.models.ppdiffusion.modules import get_time_embedder


RELU_LEAK = 0.2


class UNetBlock(paddle.nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=None,
        transposed=False,
        bn=True,
        relu=True,
        size=4,
        pad=1,
        dropout=0.0,
    ):
        super().__init__()
        batch_norm = bn
        relu_leak = None if relu else RELU_LEAK
        kernal_size = size
        self.time_mlp = (
            paddle.nn.Sequential(
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=time_emb_dim, out_features=out_channels * 2),
            )
            if time_emb_dim is not None
            else None
        )

        ops = []
        # Next, the actual conv op
        if not transposed:
            # Regular conv
            ops.append(
                paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernal_size,
                    stride=2,
                    padding=pad,
                    bias_attr=True,
                )
            )
        else:
            # Upsample and transpose conv
            ops.append(paddle.nn.Upsample(scale_factor=2, mode="bilinear"))
            ops.append(
                paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernal_size - 1,
                    stride=1,
                    padding=pad,
                    bias_attr=True,
                )
            )
        # Finally, optional batch norm
        if batch_norm:
            ops.append(paddle.nn.BatchNorm2D(num_features=out_channels))
        else:
            ops.append(paddle.nn.GroupNorm(num_groups=8, num_channels=out_channels))

        # Bundle ops into Sequential
        self.ops = paddle.nn.Sequential(*ops)

        # First the activation
        if relu_leak is None or relu_leak == 0:
            self.act = paddle.nn.ReLU()
        else:
            self.act = paddle.nn.LeakyReLU(negative_slope=relu_leak)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x, time_emb=None):
        x = self.ops(x)
        if self.time_mlp is not None:
            assert time_emb is not None, "Time embedding must be provided if time_mlp is not None"
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(chunks=2, axis=1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)
        return x


class UNet(BaseModel):
    def __init__(
        self,
        dim: int,
        num_input_channels: int,
        num_output_channels: int,
        num_cond_channels: int = 0,
        with_time_emb: bool = False,
        outer_sample_mode: str = "bilinear",  # bilinear or nearest
        upsample_dims: tuple = (256, 256),  # (256, 256) or (128, 128)
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_input_channels, num_output_channels, num_cond_channels)
        self.outer_sample_mode = outer_sample_mode
        if upsample_dims is None:
            self.upsampler = paddle.nn.Identity()
        else:
            self.upsampler = paddle.nn.Upsample(size=tuple(upsample_dims), mode=self.outer_sample_mode)
        in_channels = self.num_input_channels + self.num_cond_channels
        if with_time_emb:
            self.time_dim = dim * 2
            self.time_emb_mlp = get_time_embedder(self.time_dim, dim, learned_sinusoidal_cond=False)
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        block_kwargs = dict(time_emb_dim=self.time_dim, dropout=dropout)

        # ENCODER LAYERS
        self.init_conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True,
        )
        self.dropout_input = paddle.nn.Dropout(p=input_dropout)
        in_channels = [dim, dim * 2, dim * 2, dim * 4, dim * 8, dim * 8]
        out_channels = [dim * 2, dim * 2, dim * 4, dim * 8, dim * 8, dim * 8]
        bns = [True, True, True, True, True, False]
        optional_params = [{}, {}, {}, {"size": 4}, {"size": 2, "pad": 0}, {"size": 2, "pad": 0}]
        encoder_kwargs = {"transposed": False, "relu": False}
        encoder_kwargs.update(block_kwargs)
        self.input_ops = paddle.nn.LayerList()
        for i, (in_ch, out_ch, bn, opt) in enumerate(zip(in_channels, out_channels, bns, optional_params)):
            current_kwargs = {**encoder_kwargs}
            current_kwargs["bn"] = bn
            if "size" in opt:
                current_kwargs["size"] = opt["size"]
            if "pad" in opt:
                current_kwargs["pad"] = opt["pad"]
            self.input_ops.append(
                UNetBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    **current_kwargs,
                )
            )

        # DECODER LAYERS
        in_channels = [dim * 8, dim * 8 * 2, dim * 8 * 2, dim * 4 * 2, dim * 2 * 2, dim * 2 * 2]
        out_channels = [dim * 8, dim * 8, dim * 4, dim * 2, dim * 2, dim]
        decoder_kwargs = {"transposed": True, "bn": True, "relu": True}
        decoder_kwargs.update(block_kwargs)
        self.output_ops = paddle.nn.LayerList()
        for i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            current_kwargs = {**decoder_kwargs}
            if i == 0 or i == 1:
                current_kwargs["size"] = 2
                current_kwargs["pad"] = 0
            self.output_ops.append(
                UNetBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    **current_kwargs,
                )
            )
        self.readout = paddle.nn.Sequential(
            paddle.nn.Conv2DTranspose(
                in_channels=dim,
                out_channels=self.num_output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias_attr=True,
            )
        )

        # Initialize weights
        self.apply(self.__init_weights)

    def __init_weights(self, layer):
        if isinstance(layer, (paddle.nn.Conv2D, paddle.nn.Conv2DTranspose)):
            layer.weight.set_value(paddle.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
        elif isinstance(layer, paddle.nn.BatchNorm2D):
            layer.weight.set_value(paddle.normal(mean=1.0, std=0.02, shape=layer.weight.shape))
            layer.bias.set_value(paddle.zeros(shape=layer.bias.shape))

    def _apply_ops(self, x: paddle.Tensor, time: paddle.Tensor = None):
        skip_connections = []
        # Encoder ops
        x = self.init_conv(x)
        x = self.dropout_input(x)
        for op in self.input_ops:
            x = op(x, time)
            skip_connections.append(x)
        # Decoder ops
        x = skip_connections.pop()
        for op in self.output_ops:
            x = op(x, time)
            if skip_connections:
                x = paddle.concat(x=[x, skip_connections.pop()], axis=1)
        x = self.readout(x)
        return x

    def forward(self, inputs, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        # Preprocess inputs for shape
        if self.num_cond_channels > 0:
            x = paddle.concat(x=[inputs, condition], axis=1)
        else:
            x = inputs
            assert condition is None

        t = self.time_emb_mlp(time) if self.time_emb_mlp is not None else None

        # Apply operations
        orig_x_shape = tuple(x.shape)[-2:]
        x = self.upsampler(x)
        y = self._apply_ops(x, t)
        y = paddle.nn.functional.interpolate(x=y, size=orig_x_shape, mode=self.outer_sample_mode)
        return y
