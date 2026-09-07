import paddle

from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_zero_bias)


class LinearProjection(paddle.nn.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        ndim=None,
        bias=True,
        optional=False,
        init_weights="xavier_uniform",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.bias = bias
        self.init_weights = init_weights
        if optional and input_dim == output_dim:
            self.proj = paddle.nn.Identity()
        elif ndim is None:
            self.proj = paddle.nn.Linear(input_dim, output_dim, bias_attr=bias)
        elif ndim == 1:
            self.proj = paddle.nn.Conv1d(
                input_dim, output_dim, kernel_size=1, bias=bias
            )
        elif ndim == 2:
            self.proj = paddle.nn.Conv2d(
                input_dim, output_dim, kernel_size=1, bias=bias
            )
        elif ndim == 3:
            self.proj = paddle.nn.Conv3d(
                input_dim, output_dim, kernel_size=1, bias=bias
            )
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "paddle":
            pass
        elif self.init_weights == "xavier_uniform":
            init_xavier_uniform_zero_bias(self.proj)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            init_truncnormal_zero_bias(self.proj)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.proj(x)
