import paddle

from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_zero_bias)
from kappamodules.utils.param_checking import to_2tuple


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_dim,
        hidden_dim=None,
        out_dim=None,
        act_ctor=paddle.nn.GELU,
        bias=True,
        init_weights="xavier_uniform",
        init_last_proj_zero=False,
    ):
        super().__init__()
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        bias1, bias2 = to_2tuple(bias)
        self.fc1 = paddle.nn.Linear(in_dim, hidden_dim, bias_attr=bias1)
        self.act = act_ctor()
        self.fc2 = paddle.nn.Linear(hidden_dim, out_dim, bias_attr=bias2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            paddle.nn.init.zeros_(self.fc2.weight)
            if self.fc2.bias is not None:
                paddle.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
