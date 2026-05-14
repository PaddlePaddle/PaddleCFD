import math

import paddle

from .grid import grid


class GConv2d(paddle.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        first_layer=False,
        last_layer=False,
        spectral=False,
        Hermitian=False,
        reflection=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = paddle.complex64 if spectral else paddle.float32
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = paddle.nn.Parameter(
                paddle.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    dtype=dtype,
                )
            )
        elif self.Hermitian:
            self.W = paddle.nn.ParameterDict(
                parameters={
                    "y0_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels,
                            1,
                            in_channels,
                            self.group_size,
                            self.kernel_size_X - 1,
                            1,
                            2,
                            dtype=paddle.float32,
                        )
                    ),
                    "yposx_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels,
                            1,
                            in_channels,
                            self.group_size,
                            self.kernel_size_Y,
                            self.kernel_size_X - 1,
                            2,
                            dtype=paddle.float32,
                        )
                    ),
                    "00_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels,
                            1,
                            in_channels,
                            self.group_size,
                            1,
                            1,
                            dtype=paddle.float32,
                        )
                    ),
                }
            )
        else:
            self.W = paddle.nn.Parameter(
                paddle.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.group_size,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    dtype=dtype,
                )
            )
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = (
            paddle.nn.Parameter(paddle.empty(1, out_channels, 1, 1)) if bias else None
        )
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for key in self.W:
                paddle.nn.init.kaiming_uniform_(self.W[key], a=math.sqrt(5))
        else:
            paddle.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            paddle.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return
        if self.Hermitian:
            y0_modes = paddle.as_complex(self.W["y0_modes"])
            yposx_modes = paddle.as_complex(self.W["yposx_modes"])
            self.weights = paddle.cat(
                [
                    y0_modes,
                    self.W["00_modes"].astype(paddle.complex64),
                    y0_modes.flip(axis=(-2,)).conj(),
                ],
                dim=-2,
            )
            self.weights = paddle.cat([self.weights, yposx_modes], dim=-1)
            self.weights = paddle.cat(
                [self.weights[..., 1:].conj().rot90(k=2, axes=[-2, -1]), self.weights],
                dim=-1,
            )
        else:
            self.weights = self.W[:]
        if self.first_layer or self.last_layer:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, axes=[-2, -1])
            if self.reflection:
                self.weights[:, self.rt_group_size :] = self.weights[
                    :, : self.rt_group_size
                ].flip(axis=[-2])
            if self.first_layer:
                self.weights = self.weights.view(
                    -1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y
                )
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(
                    self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y
                )
                self.bias = self.B
        else:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(axes=[-2, -1])
                if self.reflection:
                    self.weights[:, k] = paddle.cat(
                        [
                            self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                            self.weights[:, k, :, : self.rt_group_size - 1],
                            self.weights[:, k, :, self.rt_group_size + 1 :],
                            self.weights[:, k, :, self.rt_group_size].unsqueeze(2),
                        ],
                        dim=2,
                    )
                else:
                    self.weights[:, k] = paddle.cat(
                        [
                            self.weights[:, k, :, -1].unsqueeze(2),
                            self.weights[:, k, :, :-1],
                        ],
                        dim=2,
                    )
            if self.reflection:
                self.weights[:, self.rt_group_size :] = paddle.cat(
                    [
                        self.weights[:, : self.rt_group_size, :, self.rt_group_size :],
                        self.weights[:, : self.rt_group_size, :, : self.rt_group_size],
                    ],
                    dim=3,
                ).flip(axis=[-2])
            self.weights = self.weights.view(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_Y,
                self.kernel_size_Y,
            )
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X :]

    def forward(self, x):
        self.get_weight()
        x = paddle.nn.functional.conv2d(input=x, weight=self.weights)
        if self.B is not None:
            x = x + self.bias
        return x


class GSpectralConv2d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection=False):
        super(GSpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.conv = GConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * modes - 1,
            reflection=reflection,
            bias=False,
            spectral=True,
            Hermitian=True,
        )
        self.get_weight()

    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    def compl_mul2d(self, input, weights):
        return paddle.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        freq0_y = (
            (paddle.fft.fftshift(paddle.fft.fftfreq(n=x.shape[-2])) == 0)
            .nonzero()
            .item()
        )
        self.get_weight()
        x_ft = paddle.fft.fftshift(paddle.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., freq0_y - self.modes + 1 : freq0_y + self.modes, : self.modes]
        out_ft = paddle.zeros(
            batchsize,
            self.weights.shape[0],
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[
            ..., freq0_y - self.modes + 1 : freq0_y + self.modes, : self.modes
        ] = self.compl_mul2d(x_ft, self.weights)
        x = paddle.fft.irfft2(
            paddle.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1))
        )
        return x


class GMLP2d(paddle.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        reflection=False,
        last_layer=False,
    ):
        super(GMLP2d, self).__init__()
        self.mlp1 = GConv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            reflection=reflection,
        )
        self.mlp2 = GConv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = paddle.nn.functional.gelu(x)
        x = self.mlp2(x)
        return x


class GNorm(paddle.nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = paddle.nn.InstanceNorm3D(num_features=width, weight_attr=False, bias_attr=False)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x


class GFNO2d(paddle.nn.Module):
    def __init__(self, num_channels, modes, width, initial_step, reflection, grid_type):
        super(GFNO2d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.modes = modes
        self.width = width
        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = GConv2d(
            in_channels=num_channels * initial_step + self.grid.grid_dim,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
            first_layer=True,
        )
        self.conv0 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv1 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv2 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv3 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.mlp0 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp1 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp2 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp3 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.w0 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w1 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w2 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w3 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))
        self.q = GMLP2d(
            in_channels=self.width,
            out_channels=num_channels,
            mid_channels=self.width * 4,
            reflection=reflection,
            last_layer=True,
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = x.permute(0, 3, 1, 2)
        x = self.p(x)
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x)
        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x)
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x)
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)


class GConv3d(paddle.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernel_size_T,
        bias=True,
        first_layer=False,
        last_layer=False,
        spectral=False,
        Hermitian=False,
        reflection=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = paddle.complex64 if spectral else paddle.float32
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.kernel_size_T_full = kernel_size_T
        self.kernel_size_T = kernel_size_T // 2 + 1 if Hermitian else kernel_size_T
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = paddle.nn.Parameter(
                paddle.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    self.kernel_size_T,
                    dtype=dtype,
                )
            )
        elif self.Hermitian:
            self.W = paddle.nn.ParameterDict(
                parameters={
                    "y00_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels,
                            1,
                            in_channels,
                            self.group_size,
                            self.kernel_size_X - 1,
                            1,
                            1,
                            dtype=paddle.complex64,
                        )
                    ),
                    "yposx0_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels,
                            1,
                            in_channels,
                            self.group_size,
                            self.kernel_size_Y,
                            self.kernel_size_X - 1,
                            1,
                            dtype=paddle.complex64,
                        )
                    ),
                    "000_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels, 1, in_channels, self.group_size, 1, 1, 1
                        )
                    ),
                    "yxpost_modes": paddle.nn.Parameter(
                        paddle.empty(
                            out_channels,
                            1,
                            in_channels,
                            self.group_size,
                            self.kernel_size_Y,
                            self.kernel_size_Y,
                            self.kernel_size_T - 1,
                            dtype=paddle.complex64,
                        )
                    ),
                }
            )
        else:
            self.W = paddle.nn.Parameter(
                paddle.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.group_size,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    self.kernel_size_T,
                    dtype=dtype,
                )
            )
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = (
            paddle.nn.Parameter(paddle.empty(1, out_channels, 1, 1, 1))
            if bias
            else None
        )
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for key in self.W:
                paddle.nn.init.kaiming_uniform_(self.W[key], a=math.sqrt(5))
        else:
            paddle.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            paddle.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return
        if self.Hermitian:
            self.weights = paddle.cat(
                [
                    self.W["y00_modes"].conj().flip(axis=(-3,)),
                    self.W["000_modes"].astype(paddle.complex64),
                    self.W["y00_modes"],
                ],
                dim=-3,
            )
            self.weights = paddle.cat(
                [
                    self.W["yposx0_modes"].conj().rot90(k=2, axes=[-3, -2]),
                    self.weights,
                    self.W["yposx0_modes"],
                ],
                dim=-2,
            )
            self.weights = paddle.cat(
                [
                    self.W["yxpost_modes"]
                    .conj()
                    .rot90(k=2, axes=[-3, -2])
                    .flip(axis=(-1,)),
                    self.weights,
                    self.W["yxpost_modes"],
                ],
                dim=-1,
            )
        else:
            self.weights = self.W[:]
        if self.first_layer or self.last_layer:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, axes=[-3, -2])
            if self.reflection:
                self.weights[:, self.rt_group_size :] = self.weights[
                    :, : self.rt_group_size
                ].flip(axis=[-3])
            if self.first_layer:
                self.weights = self.weights.view(
                    -1,
                    self.in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_Y,
                    self.kernel_size_T,
                )
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(
                    self.out_channels,
                    -1,
                    self.kernel_size_Y,
                    self.kernel_size_Y,
                    self.kernel_size_T,
                )
                self.bias = self.B
        else:
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(axes=[-3, -2])
                if self.reflection:
                    self.weights[:, k] = paddle.cat(
                        [
                            self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                            self.weights[:, k, :, : self.rt_group_size - 1],
                            self.weights[:, k, :, self.rt_group_size + 1 :],
                            self.weights[:, k, :, self.rt_group_size].unsqueeze(2),
                        ],
                        dim=2,
                    )
                else:
                    self.weights[:, k] = paddle.cat(
                        [
                            self.weights[:, k, :, -1].unsqueeze(2),
                            self.weights[:, k, :, :-1],
                        ],
                        dim=2,
                    )
            if self.reflection:
                self.weights[:, self.rt_group_size :] = paddle.cat(
                    [
                        self.weights[:, : self.rt_group_size, :, self.rt_group_size :],
                        self.weights[:, : self.rt_group_size, :, : self.rt_group_size],
                    ],
                    dim=3,
                ).flip(axis=[-3])
            self.weights = self.weights.view(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_Y,
                self.kernel_size_Y,
                self.kernel_size_T_full,
            )
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_T :]

    def forward(self, x):
        self.get_weight()
        x = paddle.nn.functional.conv3d(input=x, weight=self.weights)
        if self.B is not None:
            x = x + self.bias
        return x


class GSpectralConv3d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes, time_modes, reflection):
        super(GSpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.time_modes = time_modes
        self.conv = GConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * modes - 1,
            kernel_size_T=2 * time_modes - 1,
            reflection=reflection,
            bias=False,
            spectral=True,
            Hermitian=True,
        )
        self.get_weight()

    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    def compl_mul3d(self, input, weights):
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        freq0_x = (
            (paddle.fft.fftshift(paddle.fft.fftfreq(n=x.shape[-2])) == 0)
            .nonzero()
            .item()
        )
        freq0_y = (
            (paddle.fft.fftshift(paddle.fft.fftfreq(n=x.shape[-3])) == 0)
            .nonzero()
            .item()
        )
        self.get_weight()
        x_ft = paddle.fft.fftshift(paddle.fft.rfftn(x, dim=[-3, -2, -1]), dim=[-3, -2])
        x_ft = x_ft[
            ...,
            freq0_y - self.modes + 1 : freq0_y + self.modes,
            freq0_x - self.modes + 1 : freq0_x + self.modes,
            : self.time_modes,
        ]
        out_ft = paddle.zeros(
            batchsize,
            self.weights.shape[0],
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[
            ...,
            freq0_y - self.modes + 1 : freq0_y + self.modes,
            freq0_x - self.modes + 1 : freq0_x + self.modes,
            : self.time_modes,
        ] = self.compl_mul3d(x_ft, self.weights)
        x = paddle.fft.irfftn(
            paddle.fft.ifftshift(out_ft, dim=[-3, -2]),
            s=(x.size(-3), x.size(-2), x.size(-1)),
        )
        return x


class GMLP3d(paddle.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        reflection=False,
        last_layer=False,
    ):
        super(GMLP3d, self).__init__()
        self.mlp1 = GConv3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
        )
        self.mlp2 = GConv3d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = paddle.nn.functional.gelu(x)
        x = self.mlp2(x)
        return x


class GFNO3d(paddle.nn.Module):
    def __init__(
        self,
        num_channels,
        modes,
        time_modes,
        width,
        initial_step,
        reflection,
        grid_type,
        time_pad=False,
    ):
        super(GFNO3d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.modes = modes
        self.time_modes = time_modes
        self.width = width
        self.time_pad = time_pad
        self.padding = 6
        self.grid = grid(twoD=False, grid_type=grid_type)
        self.p = GConv3d(
            in_channels=num_channels * initial_step + self.grid.grid_dim,
            out_channels=self.width,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
            first_layer=True,
        )
        self.conv0 = GSpectralConv3d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            time_modes=self.time_modes,
            reflection=reflection,
        )
        self.conv1 = GSpectralConv3d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            time_modes=self.time_modes,
            reflection=reflection,
        )
        self.conv2 = GSpectralConv3d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            time_modes=self.time_modes,
            reflection=reflection,
        )
        self.conv3 = GSpectralConv3d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            time_modes=self.time_modes,
            reflection=reflection,
        )
        self.mlp0 = GMLP3d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp1 = GMLP3d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp2 = GMLP3d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp3 = GMLP3d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.w0 = GConv3d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
        )
        self.w1 = GConv3d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
        )
        self.w2 = GConv3d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
        )
        self.w3 = GConv3d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            kernel_size_T=1,
            reflection=reflection,
        )
        self.q = GMLP3d(
            in_channels=self.width,
            out_channels=num_channels,
            mid_channels=self.width * 4,
            reflection=reflection,
            last_layer=True,
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        x = self.grid(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.p(x)
        if self.time_pad:
            x = paddle.compat.nn.functional.pad(x, [0, self.padding])
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x)
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x)
        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = paddle.nn.functional.gelu(x)
        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        if self.time_pad:
            x = x[..., : -self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x.unsqueeze(-2)
