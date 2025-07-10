import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation


class ResNetBlock(paddle.nn.Layer):

    def __init__(self, in_size, mid_size, activation, dtype=None):
        super(ResNetBlock, self).__init__()
        net = []
        net.append(paddle.nn.Linear(in_features=in_size, out_features=mid_size))
        net.append(activation)
        net.append(paddle.nn.Linear(in_features=mid_size, out_features=mid_size))
        net.append(activation)
        if in_size != mid_size:
            self.shortcut = paddle.nn.Sequential(paddle.nn.Linear(in_features=in_size, out_features=mid_size))
        else:
            self.shortcut = paddle.nn.Sequential()
        self.layer = paddle.nn.Sequential(*net)

    def forward(self, x):
        out = self.layer(x) + self.shortcut(x)
        return out


class ResNet(paddle.nn.Layer):

    def __init__(self, layers_list, activation="Tanh", dtype=None):
        super(ResNet, self).__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:-1]:
            net.append(ResNetBlock(self.hidden_in, hidden, self.activation, dtype=dtype))
            net.append(self.activation)
            self.hidden_in = hidden
        net.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=layers_list[-1]))
        self.net = paddle.nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
