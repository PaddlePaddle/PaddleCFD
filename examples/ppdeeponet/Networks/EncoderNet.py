import paddle


try:
    from CNNet import CNNet1d
    from CNNet import CNNet2d
    from FCNet import FCNet
    from FunActivation import FunActivation
except ImportError:
    from .CNNet import CNNet1d
    from .CNNet import CNNet2d
    from .FCNet import FCNet
    from .FunActivation import FunActivation


class EncoderFCNet(paddle.nn.Layer):

    def __init__(self, layers_list: list, activation, dtype=None) -> None:
        super(EncoderFCNet, self).__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        self.net = FCNet(layers_list, activation, dtype)

    def forward(self, x):
        """
        Input:
            x: size(n_batch, my*mx, in_size)
        Return:
            beta: size(n_batch, n_latent)
        """
        x = x.reshape(tuple(x.shape)[0], -1)
        x = self.net(x)
        return x


class EncoderFCNet_VAE(paddle.nn.Layer):

    def __init__(self, layers_list: list, activation, dtype=None) -> None:
        super(EncoderFCNet_VAE, self).__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        self.net_mu = FCNet(layers_list, activation, dtype)
        self.net_log_var = FCNet(layers_list, activation, dtype)

    def reparam(self, mu, log_var):
        """ """
        std = paddle.exp(x=0.5 * log_var)
        eps = paddle.randn(shape=log_var.shape, dtype=log_var.dtype)
        return mu + std * eps

    def forward(self, x):
        """
        Input:
            x: size(n_batch, my*mx, in_size)
        Return:
            beta: size(n_batch, n_latent)
        """
        x = x.reshape(tuple(x.shape)[0], -1)
        mu = self.net_mu(x)
        log_var = self.net_log_var(x)
        return self.reparam(mu, log_var), mu, log_var


class EncoderCNNet1d(paddle.nn.Layer):

    def __init__(
        self, conv_arch: list, fc_arch: list, activation_conv, activation_fc, dtype=None, kernel_size=5, stride=3
    ) -> None:
        super(EncoderCNNet1d, self).__init__()
        self.in_channel = conv_arch[0]
        self.net = CNNet1d(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv=activation_conv,
            activation_fc=activation_fc,
            kernel_size=kernel_size,
            stride=stride,
            dtype=dtype,
        )

    def forward(self, x):
        """
        Input:
            x: size(n_batch, mesh_size, in_channel)
        Return:
            beta: size(n_batch, n_latent)
        """
        x = x.transpose(perm=[0, 2, 1])
        x = self.net(x)
        return x


class EncoderCNNet2d(paddle.nn.Layer):

    def __init__(
        self,
        conv_arch: list,
        fc_arch: list,
        activation_conv,
        activation_fc,
        nx_size: int,
        ny_size: int,
        dtype=None,
        kernel_size=(5, 5),
        stride=3,
    ) -> None:
        super(EncoderCNNet2d, self).__init__()
        self.in_channel = conv_arch[0]
        self.nx_size = nx_size
        self.ny_size = ny_size
        self.net = CNNet2d(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv=activation_conv,
            activation_fc=activation_fc,
            kernel_size=kernel_size,
            stride=stride,
            dtype=dtype,
        )

    def forward(self, x):
        """
        Input:
            x: size(n_batch, my*mx, in_channel)
        Return:
            beta: size(n_batch, n_latent)
        """
        x = x.transpose(perm=[0, 2, 1])
        x = x.reshape([-1, self.in_channel, self.ny_size, self.nx_size])
        x = self.net(x)
        return x
