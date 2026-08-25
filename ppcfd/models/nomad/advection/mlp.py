import paddle
import paddle.nn as nn


class MLP(nn.Layer):
    def __init__(self, layers):
        super().__init__()

        layer_list = []

        for i in range(len(layers) - 2):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            layer_list.append(nn.GELU())

        layer_list.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):

        orig_shape = x.shape
        x = paddle.reshape(x, [-1, orig_shape[-1]])

        x = self.net(x)

        out_dim = x.shape[-1]

        x = paddle.reshape(x, list(orig_shape[:-1]) + [out_dim])

        return x