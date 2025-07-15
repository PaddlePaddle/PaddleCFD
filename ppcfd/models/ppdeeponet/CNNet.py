import numpy as np
import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation


class CNNet1d(paddle.nn.Layer):

    def __init__(
        self,
        conv_arch: list,
        fc_arch: list,
        activation_conv: str = "Tanh",
        activation_fc: str = "Tanh",
        kernel_size=5,
        stride: int = 3,
        dtype=None,
    ):
        super(CNNet1d, self).__init__()
        if isinstance(activation_conv, str):
            self.activation_conv = FunActivation()(activation_conv)
        else:
            self.activation_conv = activation_conv
        if isinstance(activation_fc, str):
            self.activation_fc = FunActivation()(activation_fc)
        else:
            self.activation_fc = activation_fc
        net = []
        self.arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            net.append(
                paddle.nn.Conv1D(in_channels=self.arch_in, out_channels=arch, kernel_size=kernel_size, stride=stride)
            )
            self.arch_in = arch
        self.conv_net = paddle.nn.Sequential(*net)
        net = []
        self.arch_in = fc_arch[0]
        for arch in fc_arch[1:]:
            net.append(paddle.nn.Linear(in_features=self.arch_in, out_features=arch))
            self.arch_in = arch
        self.fc_net = paddle.nn.Sequential(*net)

    def forward(self, x):
        """
        Input:
            x: size(batch_size, conv_arch[0], m_size)
        Return:
            out: size(batch_size, fc_arch[-1])
        """
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation_conv(x)
        x = x.reshape(tuple(x.shape)[0], -1)
        for fc in self.fc_net[:-1]:
            x = fc(x)
            x = self.activation_fc(x)
        x = self.fc_net[-1](x)
        return x


class CNNPure2d(paddle.nn.Layer):

    def __init__(self, conv_arch: list, activation: str = "Tanh", kernel_size=(3, 3), stride: int = 2, dtype=None):
        super(CNNPure2d, self).__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        net = []
        self.arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            net.append(
                paddle.nn.Conv2D(in_channels=self.arch_in, out_channels=arch, kernel_size=kernel_size, stride=stride)
            )
            self.arch_in = arch
        self.conv_net = paddle.nn.Sequential(*net)

    def forward(self, x):
        """
        Input:
            x: size(batch_size, conv_arch[0], my_size, mx_size)
        Return:
            out: size(batch_size, conv_arch[-1], my_szie, mx_size)
        """
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation(x)
        return x


class CNNet2d(paddle.nn.Layer):

    def __init__(
        self,
        conv_arch: list,
        fc_arch: list,
        activation_conv: str = "Tanh",
        activation_fc: str = "Tanh",
        kernel_size=(5, 5),
        stride: int = 3,
        dtype=None,
    ):
        super(CNNet2d, self).__init__()
        if isinstance(activation_conv, str):
            self.activation_conv = FunActivation()(activation_conv)
        else:
            self.activation_conv = activation_conv
        if isinstance(activation_fc, str):
            self.activation_fc = FunActivation()(activation_fc)
        else:
            self.activation_fc = activation_fc
        net = []
        self.arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            net.append(
                paddle.nn.Conv2D(in_channels=self.arch_in, out_channels=arch, kernel_size=kernel_size, stride=stride)
            )
            self.arch_in = arch
        self.conv_net = paddle.nn.Sequential(*net)
        net = []
        self.arch_in = fc_arch[0]
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / self.arch_in), high=np.sqrt(1 / self.arch_in))
        )
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / self.arch_in), high=np.sqrt(1 / self.arch_in))
        )
        for arch in fc_arch[1:]:
            net.append(
                paddle.nn.Linear(
                    in_features=self.arch_in, out_features=arch, weight_attr=weight_attr, bias_attr=bias_attr
                )
            )
            self.arch_in = arch
        self.fc_net = paddle.nn.Sequential(*net)

    def forward(self, x):
        """
        Input:
            x: size(batch_size, conv_arch[0], my_size, mx_size)
        Return:
            out: size(batch_size, fc_arch[-1])
        """
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation_conv(x)
        x = x.reshape([tuple(x.shape)[0], -1])
        for fc in self.fc_net[:-1]:
            x = fc(x)
            x = self.activation_fc(x)
        x = self.fc_net[-1](x)

        return x
