import paddle


class FreqLinear(paddle.nn.Layer):
    def __init__(self, in_channel, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = paddle.nn.Parameter(
            scale * paddle.randn(in_channel, 4 * modes1 * modes2, dtype=paddle.float32)
        )
        self.bias = paddle.nn.Parameter(
            paddle.zeros(1, 4 * modes1 * modes2, dtype=paddle.float32)
        )

    def forward(self, x):
        B = x.shape[0]
        h = paddle.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return paddle.view_as_complex(h)


def batchmul2d(x, weights, emb):
    temp = x * emb.unsqueeze(1)
    out = paddle.einsum("bixy,ioxy->boxy", temp, weights)
    return out


class SpectralConv2d(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, cond_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        @author: Zongyi Li
        [paper](https://arxiv.org/pdf/2010.08895.pdf)
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
                2,
                dtype=paddle.float32,
            )
        )
        self.weights2 = paddle.nn.Parameter(
            self.scale
            * paddle.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                2,
                dtype=paddle.float32,
            )
        )
        self.cond_emb = FreqLinear(cond_channels, self.modes1, self.modes2)

    def forward(self, x, emb):
        emb12 = self.cond_emb(emb)
        emb1 = emb12[..., 0]
        emb2 = emb12[..., 1]
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
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            paddle.view_as_complex(self.weights1),
            emb1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            paddle.view_as_complex(self.weights2),
            emb2,
        )
        x = paddle.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class EmbedSequential(paddle.nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb)
        return x


class FourierBasicBlock(paddle.nn.Layer):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        cond_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = paddle.nn.GELU()
        assert not norm
        self.fourier1 = SpectralConv2d(
            in_planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2
        )
        self.conv1 = paddle.nn.Conv2d(
            in_planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True
        )
        self.cond_emb = paddle.compat.nn.Linear(cond_channels, planes)
        self.fourier2 = SpectralConv2d(
            planes, planes, cond_channels, modes1=self.modes1, modes2=self.modes2
        )
        self.conv2 = paddle.nn.Conv2d(
            planes, planes, kernel_size=1, padding=0, padding_mode="zeros", bias=True
        )

    def forward(self, x, emb):
        x1 = self.fourier1(x, emb)
        x2 = self.conv1(x)
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(x2.shape):
            emb_out = emb_out[..., None]
        out = self.activation(x1 + x2 + emb_out)
        x1 = self.fourier2(out, emb)
        x2 = self.conv2(out)
        out = x1 + x2
        out = self.activation(out)
        return out


class ResNet(paddle.nn.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cond_dim: int,
        block: paddle.nn.Layer,
        num_blocks: list,
        norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.activation = paddle.nn.GELU()
        self.conv_in1 = paddle.nn.Conv2d(
            input_dim, hidden_dim, kernel_size=1, bias=True
        )
        self.conv_in2 = paddle.nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, bias=True
        )
        self.conv_out1 = paddle.nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, bias=True
        )
        self.conv_out2 = paddle.nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, bias=True
        )
        self.layers = paddle.nn.LayerList(
            [
                self._make_layer(
                    block=block,
                    planes=hidden_dim,
                    cond_channels=cond_dim,
                    num_blocks=num_blocks[i],
                    stride=1,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )

    @staticmethod
    def _make_layer(
        block: paddle.nn.Layer,
        planes: int,
        cond_channels: int,
        num_blocks: int,
        stride: int,
        norm: bool = True,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for _ in strides:
            layers.append(block(planes, planes, cond_channels=cond_channels, norm=norm))
        return EmbedSequential(*layers)

    def __repr__(self):
        return "ResNet"

    def forward(self, x, cond):
        assert x.dim() == 4
        x = self.activation(self.conv_in1(x))
        x = self.activation(self.conv_in2(x))
        for layer in self.layers:
            x = layer(x, cond)
        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)
        return x
