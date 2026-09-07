import paddle

from .FNO import MLP2d, MLP3d
from .grid import grid


class radialSpectralConv2d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection):
        super(radialSpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.reflection = reflection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.dtype = paddle.float32
        if reflection:
            self.inds_lower = paddle.tril_indices(
                row=self.modes + 1, col=self.modes + 1
            )
            self.W = paddle.nn.Parameter(
                self.scale
                * paddle.rand(
                    in_channels,
                    out_channels,
                    self.inds_lower.shape[1],
                    dtype=self.dtype,
                )
            )
        else:
            self.W_LC = paddle.nn.Parameter(
                self.scale
                * paddle.rand(
                    in_channels, out_channels, self.modes + 1, 1, dtype=self.dtype
                )
            )
            self.W_LR = paddle.nn.Parameter(
                self.scale
                * paddle.rand(
                    in_channels, out_channels, self.modes, self.modes, dtype=self.dtype
                )
            )
        self.eval_build = True
        self.get_weight()

    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return
        if self.reflection:
            W_LR = paddle.zeros(
                self.in_channels,
                self.out_channels,
                self.modes + 1,
                self.modes + 1,
                dtype=self.dtype,
            ).to(self.W.device)
            W_LR[..., self.inds_lower[0], self.inds_lower[1]] = self.W
            W_LR.transpose(-1, -2)[..., self.inds_lower[0], self.inds_lower[1]] = self.W
            self.weights = paddle.cat(
                [W_LR[..., 1:, :].flip(axis=-2), W_LR], dim=-2
            ).cfloat()
        else:
            W_LR = paddle.cat([self.W_LC[:, :, 1:], self.W_LR], dim=-1)
            W_UR = paddle.cat(
                [self.W_LC.flip(axis=-2), W_LR.rot90(axes=[-2, -1])], dim=-1
            )
            self.weights = paddle.cat([W_UR, W_LR], dim=-2).cfloat()

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
        x_ft = x_ft[
            ..., freq0_y - self.modes : freq0_y + self.modes + 1, : self.modes + 1
        ]
        out_ft = paddle.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[
            ..., freq0_y - self.modes : freq0_y + self.modes + 1, : self.modes + 1
        ] = self.compl_mul2d(x_ft, self.weights)
        x = paddle.fft.irfft2(
            paddle.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1))
        )
        return x


class radialNO2d(paddle.nn.Module):
    def __init__(self, num_channels, modes, width, initial_step, reflection, grid_type):
        super(radialNO2d, self).__init__()
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
        self.act = paddle.nn.ReLU()
        self.norm = paddle.nn.InstanceNorm2D(num_features=width)
        self.modes = modes
        self.width = width
        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = paddle.compat.nn.Linear(
            initial_step * num_channels + self.grid.grid_dim, self.width
        )
        self.conv0 = radialSpectralConv2d(
            self.width, self.width, self.modes, reflection
        )
        self.conv1 = radialSpectralConv2d(
            self.width, self.width, self.modes, reflection
        )
        self.conv2 = radialSpectralConv2d(
            self.width, self.width, self.modes, reflection
        )
        self.conv3 = radialSpectralConv2d(
            self.width, self.width, self.modes, reflection
        )
        self.mlp0 = MLP2d(self.width, self.width, self.width)
        self.mlp1 = MLP2d(self.width, self.width, self.width)
        self.mlp2 = MLP2d(self.width, self.width, self.width)
        self.mlp3 = MLP2d(self.width, self.width, self.width)
        self.w0 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.w1 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.w2 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.w3 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.q = MLP2d(self.width, num_channels, self.width * 4)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)
        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)


class radialSpectralConv3d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes, time_modes, reflection):
        super(radialSpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.reflection = reflection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.time_modes = time_modes
        self.scale = 1 / (in_channels * out_channels)
        self.dtype = paddle.float32
        if reflection:
            self.inds_lower = paddle.tril_indices(row=self.modes, col=self.modes)
            self.W = paddle.nn.Parameter(
                self.scale
                * paddle.rand(
                    in_channels,
                    out_channels,
                    self.inds_lower.shape[1],
                    self.time_modes,
                    dtype=self.dtype,
                )
            )
        else:
            self.W_LC = paddle.nn.Parameter(
                self.scale
                * paddle.rand(
                    in_channels,
                    out_channels,
                    self.modes,
                    1,
                    self.time_modes,
                    dtype=self.dtype,
                )
            )
            self.W_LR = paddle.nn.Parameter(
                self.scale
                * paddle.rand(
                    in_channels,
                    out_channels,
                    self.modes - 1,
                    self.modes - 1,
                    self.time_modes,
                    dtype=self.dtype,
                )
            )
        self.eval_build = True
        self.get_weight()

    def get_weight(self):
        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return
        if self.reflection:
            W_LR = paddle.zeros(
                self.in_channels,
                self.out_channels,
                self.modes,
                self.modes,
                self.time_modes,
                dtype=self.dtype,
            ).to(self.W.device)
            W_LR[..., self.inds_lower[0], self.inds_lower[1], :] = self.W
            W_LR.transpose(-2, -3)[
                ..., self.inds_lower[0], self.inds_lower[1], :
            ] = self.W
            W_R = paddle.cat([W_LR[..., 1:, :, :].flip(axis=-3), W_LR], dim=-3)
        else:
            W_LR = paddle.cat([self.W_LC[:, :, 1:], self.W_LR], dim=-2)
            W_UR = paddle.cat(
                [self.W_LC.flip(axis=-3), W_LR.rot90(axes=[-3, -2])], dim=-2
            )
            W_R = paddle.cat([W_UR, W_LR], dim=-3)
        self.weights = paddle.cat(
            [W_R[..., 1:, :].flip(axis=[-3, -2]), W_R], dim=-2
        ).cfloat()

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
            self.out_channels,
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


class radialNO3d(paddle.nn.Module):
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
        super(radialNO3d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 3 locations (u(t-10, x, y, z), ..., u(t-1, x, y, z),  x, y, z)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.act = paddle.nn.ReLU()
        self.modes = modes
        self.time_modes = time_modes
        self.width = width
        self.time_pad = time_pad
        self.padding = 6
        self.grid = grid(twoD=False, grid_type=grid_type)
        self.p = paddle.compat.nn.Linear(
            initial_step * num_channels + self.grid.grid_dim, self.width
        )
        self.conv0 = radialSpectralConv3d(
            self.width, self.width, self.modes, self.time_modes, reflection
        )
        self.conv1 = radialSpectralConv3d(
            self.width, self.width, self.modes, self.time_modes, reflection
        )
        self.conv2 = radialSpectralConv3d(
            self.width, self.width, self.modes, self.time_modes, reflection
        )
        self.conv3 = radialSpectralConv3d(
            self.width, self.width, self.modes, self.time_modes, reflection
        )
        self.mlp0 = MLP3d(self.width, self.width, self.width)
        self.mlp1 = MLP3d(self.width, self.width, self.width)
        self.mlp2 = MLP3d(self.width, self.width, self.width)
        self.mlp3 = MLP3d(self.width, self.width, self.width)
        self.w0 = paddle.nn.Conv3d(self.width, self.width, 1)
        self.w1 = paddle.nn.Conv3d(self.width, self.width, 1)
        self.w2 = paddle.nn.Conv3d(self.width, self.width, 1)
        self.w3 = paddle.nn.Conv3d(self.width, self.width, 1)
        self.q = MLP3d(self.width, num_channels, self.width * 4)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        x = self.grid(x)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        if self.time_pad:
            x = paddle.compat.nn.functional.pad(x, [0, self.padding])
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)
        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)
        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)
        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        if self.time_pad:
            x = x[..., : -self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x
