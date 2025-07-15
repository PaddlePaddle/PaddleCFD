import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation


class FCNet(paddle.nn.Layer):

    def __init__(self, layers_list: list, activation: str = "Tanh", dtype=None, kernel_init=None):
        super(FCNet, self).__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:]:
            net.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            self.hidden_in = hidden
        self.net = paddle.nn.Sequential(*net)

    def forward(self, x):
        for net in self.net[:-1]:
            x = net(x)
            x = self.activation(x)
        x = self.net[-1](x)
        return x
