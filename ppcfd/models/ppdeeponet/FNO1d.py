import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation


class SpectralConv1d(paddle.nn.Layer):

    def __init__(self, in_size: int, out_size: int, modes: int, dtype):
        super(SpectralConv1d, self).__init__()
        """1D Fourier layer: FFT -> linear transform -> Inverse FFT
        """
        self.in_size = in_size
        self.out_size = out_size
        self.modes = modes
        self.scale = 1.0 / (in_size * out_size)
        if dtype is None or dtype == "float32":
            ctype = "complex64"
        elif dtype == "float64":
            ctype = "complex128"
        else:
            raise TypeError("No such data type.")
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=self.scale * paddle.rand(shape=[in_size, out_size, self.modes], dtype=ctype)
        )

    def compl_mul_1d(self, input, weights):
        """Complex multiplication: (batch_size, in_size, m) , (in_size, out_size, m) -> (batch_size, out_size, m)"""
        return paddle.einsum("bim, iom->bom", input, weights)

    def forward(self, x):
        """
        Input:
            x: size(batch_size, in_size, mesh_size)
        """
        batch_size = tuple(x.shape)[0]
        x_ft = paddle.fft.rfft(x=x)
        out_ft = paddle.zeros(shape=[batch_size, self.out_size, x.shape[-1] // 2 + 1], dtype="complex64")
        out_ft[:, :, : self.modes] = self.compl_mul_1d(x_ft[:, :, : self.modes], self.weight)
        x = paddle.fft.irfft(x=out_ft, n=x.shape[-1])
        return x


class FNO1d(paddle.nn.Layer):

    def __init__(
        self, in_size: int, out_size: int, modes: int, hidden_list: list[int], activation: str = "ReLU", dtype=None
    ):
        super(FNO1d, self).__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        self.fc_in = paddle.nn.Linear(in_features=in_size, out_features=hidden_list[0])
        conv_net, w_net = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            conv_net.append(SpectralConv1d(self.hidden_in, hidden, modes, dtype))
            w_net.append(paddle.nn.Conv1D(in_channels=self.hidden_in, out_channels=hidden, kernel_size=1))
            self.hidden_in = hidden
        self.spectral_conv = paddle.nn.Sequential(*conv_net)
        self.weight_conv = paddle.nn.Sequential(*w_net)
        self.fc_out0 = paddle.nn.Linear(in_features=self.hidden_in, out_features=128)
        self.fc_out1 = paddle.nn.Linear(in_features=128, out_features=out_size)

    def forward(self, x):
        """
        Input:
            x: size(batch_size, mesh_size, in_size)
        Output:
            x: size(batch_size, mesh_size, out_size)
        """
        x = self.fc_in(x)
        x = x.transpose(perm=[0, 2, 1])
        for conv, weight in zip(self.spectral_conv, self.weight_conv):
            x1 = conv(x)
            x2 = weight(x)
            x = self.activation(x1 + x2)
        x = x.transpose(perm=[0, 2, 1])
        x = self.fc_out0(x)
        x = self.activation(x)
        return self.fc_out1(x)
