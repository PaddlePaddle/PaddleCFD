import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation


class SpectralConv2d(paddle.nn.Layer):

    def __init__(self, in_size, out_size, modes1, modes2, dtype):
        super(SpectralConv2d, self).__init__()
        """2D Fourier layer: FFT -> linear transform -> Inverse FFT
        """
        self.in_size = in_size
        self.out_size = out_size
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_size * out_size)
        if dtype is None or dtype == "float32":
            ctype = "complex64"
        elif dtype == "float64":
            ctype = "complex128"
        else:
            raise TypeError("No such data type.")
        self.weight1 = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=self.scale * paddle.rand(shape=[in_size, out_size, modes1, modes2], dtype=ctype)
        )
        self.weight2 = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=self.scale * paddle.rand(shape=[in_size, out_size, modes1, modes2], dtype=ctype)
        )

    def compl_mul_2d(self, input, weights):
        """Complex multiplication"""
        return paddle.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = tuple(x.shape)[0]
        x_ft = paddle.fft.rfft2(x=x)
        out_ft = paddle.zeros(shape=[batch_size, self.out_size, x.shape[-2], x.shape[-1] // 2 + 1], dtype="complex64")
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul_2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weight1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul_2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weight2
        )
        x = paddle.fft.irfft2(x=out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class FNO2d(paddle.nn.Layer):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        modes1: int,
        modes2: int,
        hidden_list: list[int],
        activation="ReLU",
        dtype=None,
    ):
        super(FNO2d, self).__init__()
        self.hidden_list = hidden_list
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        self.fc_in = paddle.nn.Linear(in_features=in_size, out_features=hidden_list[0])
        conv_net, w_net = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            conv_net.append(SpectralConv2d(self.hidden_in, hidden, modes1, modes2, dtype))
            w_net.append(paddle.nn.Conv1D(in_channels=self.hidden_in, out_channels=hidden, kernel_size=1))
            self.hidden_in = hidden
        self.spectral_conv = paddle.nn.Sequential(*conv_net)
        self.weight_conv = paddle.nn.Sequential(*w_net)
        self.fc_out0 = paddle.nn.Linear(in_features=self.hidden_in, out_features=128)
        self.fc_out1 = paddle.nn.Linear(in_features=128, out_features=out_size)

    def forward(self, x):
        batch_size = tuple(x.shape)[0]
        mx_size, my_size = tuple(x.shape)[1], tuple(x.shape)[2]
        x = self.fc_in(x)
        x = x.transpose(perm=[0, 3, 1, 2])
        for conv, weight, hidden_size in zip(self.spectral_conv, self.weight_conv, self.hidden_list):
            x1 = conv(x)
            x2 = weight(x.view(batch_size, hidden_size, -1)).view(batch_size, hidden_size, mx_size, my_size)
            x = self.activation(x1 + x2)
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.fc_out0(x)
        x = self.activation(x)
        return self.fc_out1(x)
