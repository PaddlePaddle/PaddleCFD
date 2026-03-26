import paddle

from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_merged_linear)
from kappamodules.utils.shapes import to_ndim


class Film(paddle.nn.Layer):
    def __init__(
        self, dim_cond, dim_out, channel_first=True, init_weights="xavier_uniform"
    ):
        super().__init__()
        self.dim_cond = dim_cond
        self.dim_out = dim_out
        self.channel_first = channel_first
        self.modulation = paddle.compat.nn.Linear(dim_cond, dim_out * 2)
        self.init_weights = init_weights
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            init_xavier_uniform_merged_linear(self.modulation, num_layers=2)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, x, cond):
        mod = self.modulation(cond)
        if self.channel_first:
            scale, shift = to_ndim(mod, ndim=x.ndim).chunk(2, dim=1)
        else:
            scale, shift = to_ndim(mod, ndim=x.ndim, pad_dim=1).chunk(2, dim=-1)
        return x * (scale + 1) + shift
