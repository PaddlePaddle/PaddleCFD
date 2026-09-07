import paddle

from .grid import grid
from .paddle_utils import move_to_device


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-05):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = paddle.mean(x, 0)
        self.std = paddle.std(x=x, axis=0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        x = x * std + mean
        return x

    def to(self, device):
        self.mean = move_to_device(self.mean, device)
        self.std = move_to_device(self.std, device)
        return self

    def cpu(self):
        return self.to("cpu")


class SpectralConv2d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=paddle.complex64,
            )
        )
        self.weights2 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=paddle.complex64,
            )
        )

    def compl_mul2d(self, input, weights):
        return paddle.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfft2(x)
        out_ft = paddle.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        x = paddle.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP2d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = paddle.nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = paddle.nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = paddle.nn.functional.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(paddle.nn.Module):
    def __init__(self, num_channels, modes1, modes2, width, initial_step, grid_type):
        super(FNO2d, self).__init__()
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
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8
        self.grid = grid(twoD=True, grid_type=grid_type)
        self.p = paddle.compat.nn.Linear(
            initial_step * num_channels + self.grid.grid_dim, self.width
        )
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP2d(self.width, self.width, self.width)
        self.mlp1 = MLP2d(self.width, self.width, self.width)
        self.mlp2 = MLP2d(self.width, self.width, self.width)
        self.mlp3 = MLP2d(self.width, self.width, self.width)
        self.w0 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.w1 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.w2 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.w3 = paddle.nn.Conv2d(self.width, self.width, 1)
        self.norm = paddle.nn.InstanceNorm2D(num_features=self.width)
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

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = paddle.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = paddle.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return paddle.cat((gridx, gridy), dim=-1)


class SpectralConv3d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )
        self.weights2 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )
        self.weights3 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )
        self.weights4 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=paddle.complex64,
            )
        )

    def compl_mul3d(self, input, weights):
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = paddle.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = paddle.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=paddle.complex64,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )
        x = paddle.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP3d(paddle.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3d, self).__init__()
        self.mlp1 = paddle.nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = paddle.nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = paddle.nn.functional.gelu(x)
        x = self.mlp2(x)
        return x


class FNO3d(paddle.nn.Module):
    def __init__(
        self,
        num_channels,
        modes1,
        modes2,
        modes3,
        width,
        initial_step,
        time,
        time_pad=False,
    ):
        super(FNO3d, self).__init__()
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
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.time = time
        self.time_pad = time_pad
        self.padding = 6
        self.p = paddle.compat.nn.Linear(initial_step * num_channels + 3, self.width)
        self.conv0 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv1 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv3 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
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
        grid = self.get_grid(x.shape).to(x.device)
        x = paddle.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        if self.time and self.time_pad:
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
        if self.time and self.time_pad:
            x = x[..., : -self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)
        if not self.time:
            x = x.unsqueeze(-2)
        return x

    def get_grid(self, shape):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = paddle.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        gridy = paddle.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        gridz = paddle.linspace(0, 1, size_z)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        return paddle.cat((gridx, gridy, gridz), dim=-1)
