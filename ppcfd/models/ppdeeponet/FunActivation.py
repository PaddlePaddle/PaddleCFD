import numpy as np
import paddle


class Sinc(paddle.nn.Layer):

    def __init__(self):
        super(Sinc, self).__init__()

    def forward(self, x):
        return x * paddle.sin(x=x)


class Swish(paddle.nn.Layer):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(x=x)


class Tanh_Sin(paddle.nn.Layer):

    def __init__(self):
        super(Tanh_Sin, self).__init__()

    def fun_sin(self, x):
        """ """
        return paddle.sin(x=np.pi * (x + 1.0))

    def forward(self, x):
        return paddle.nn.functional.tanh(x=self.fun_sin(x)) + x


class FunActivation:

    def __init__(self, **kwrds):
        self.activation = {
            "Identity": paddle.nn.Identity(),
            "ReLU": paddle.nn.ReLU(),
            "ELU": paddle.nn.ELU(),
            "Softplus": paddle.nn.Softplus(),
            "Sigmoid": paddle.nn.Sigmoid(),
            "Tanh": paddle.nn.Tanh(),
            "SiLU": paddle.nn.Silu(),
            "Swish": Swish(),
            "Sinc": Sinc(),
            "Tanh_Sin": Tanh_Sin(),
        }

    def __call__(self, type=str):
        return self.activation[type]
