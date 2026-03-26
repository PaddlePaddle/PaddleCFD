import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class WeightNormLinear(paddle.nn.Layer):
    """
    torch.nn.utils.weight_norm(nn.Linear(...)) but with weight_g as buffer when it is fixed
    if weight_g is set to requires_grad=False (as done in DINO, iBOT, MUGS, ...) and during
    training a parent module is unfrozen via
    ```
    for p in module.parameters():
      p.requires_grad = True
    ```
    the weight_g is also unfrozen. registering weight_g as a buffer avoids this
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fixed_g: bool = False,
        device=None,
        dtype="float32",
        init_weights="xavier_uniform",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fixed_g = fixed_g
        self.init_weights = init_weights
        self.weight_v = self.create_parameter(
            shape=[out_features, in_features],
            dtype=dtype,
            default_initializer=nn.initializer.XavierUniform() # 默认给一个初始化，后面会被 reset_parameters 覆盖
        )
        if fixed_g:
            self.register_buffer(
                "weight_g", 
                paddle.ones([out_features, 1], dtype=dtype)
            )
        else:
            self.weight_g = self.create_parameter(
                shape=[out_features, 1],
                dtype=dtype,
                default_initializer=nn.initializer.Constant(1.0)
            )
        if bias:
            self.bias = self.create_parameter(
                shape=[out_features],
                dtype=dtype,
                is_bias=True
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            init = nn.initializer.KaimingUniform(negative_slope=math.sqrt(5))
            init(self.weight_v)
            if self.bias is not None:
                fan_in = self.weight_v.shape[1]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init_b = nn.initializer.Uniform(-bound, bound)
                init_b(self.bias)

        elif self.init_weights == "xavier_uniform":
            init = nn.initializer.XavierUniform()
            init(self.weight_v)
            if self.bias is not None:
                init_b = nn.initializer.Constant(0.0)
                init_b(self.bias)
        with paddle.no_grad():
            if self.fixed_g:
                self.weight_g.set_value(paddle.ones_like(self.weight_g))
            else:
                norm = paddle.norm(self.weight_v, axis=1, keepdim=True)
                self.weight_g.set_value(norm)

    def extra_repr(self):
        return paddle.nn.Linear.extra_repr(self)

    def forward(self, x):
        norm_v = F.normalize(self.weight_v, axis=1)
        weight = self.weight_g * norm_v
        return F.linear(x, weight.t(), self.bias)