from functools import partial

import paddle

from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_zero_bias)
from kappamodules.norm.global_response_norm import GlobalResponseNorm
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
        ndim=None,
        use_global_response_norm=False,
    ):
        super().__init__()
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        bias1, bias2 = to_2tuple(bias)
        # if ndim is None:
        #     fc_ctor = paddle.compat.nn.Linear
        # elif ndim == 1:
        #     fc_ctor = partial(paddle.nn.Conv1D, kernel_size=1)
        # elif ndim == 2:
        #     fc_ctor = partial(paddle.nn.Conv2D, kernel_size=1)
        # elif ndim == 3:
        #     fc_ctor = partial(paddle.nn.Conv3D, kernel_size=1)
        # else:
        #     raise NotImplementedError
        # self.fc1 = fc_ctor(in_dim, hidden_dim, bias=bias1)
        # self.act = act_ctor()
        # if use_global_response_norm:
        #     self.grn = GlobalResponseNorm(dim=hidden_dim, ndim=ndim)
        # else:
        #     self.grn = paddle.nn.Identity()
        # self.fc2 = fc_ctor(hidden_dim, out_dim, bias=bias2)
        # self.reset_parameters()
        if ndim is None:
            # 对于 Linear 层，使用 bias 参数
            fc_ctor = paddle.compat.nn.Linear
            self.fc1 = fc_ctor(in_dim, hidden_dim, bias=bias1)
            self.fc2 = fc_ctor(hidden_dim, out_dim, bias=bias2)
        else:
            # 对于 Conv 层，使用 bias_attr 参数
            if ndim == 1:
                conv_ctor = paddle.nn.Conv1D
            elif ndim == 2:
                conv_ctor = paddle.nn.Conv2D
            elif ndim == 3:
                conv_ctor = paddle.nn.Conv3D
            else:
                raise NotImplementedError
            # 将 bias 转换为 bias_attr
            bias_attr1 = None if bias1 else False
            bias_attr2 = None if bias2 else False
            self.fc1 = conv_ctor(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1, bias_attr=bias_attr1)
            self.fc2 = conv_ctor(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, bias_attr=bias_attr2)
        self.act = act_ctor()
        if use_global_response_norm:
            self.grn = GlobalResponseNorm(dim=hidden_dim, ndim=ndim)
        else:
            self.grn = paddle.nn.Identity()
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
        x = self.grn(x)
        x = self.fc2(x)
        return x
