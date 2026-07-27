import paddle

from kappamodules.init.functional import (init_truncnormal_zero_bias,
                                          init_xavier_uniform_merged_linear,
                                          init_zero)


class Dit(paddle.nn.Layer):
    def __init__(
        self,
        cond_dim,
        out_dim,
        num_outputs=6,
        gate_indices=None,
        init_weights="xavier_uniform",
        init_gate_zero=False,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.num_outputs = num_outputs
        self.modulation = paddle.compat.nn.Linear(cond_dim, num_outputs * out_dim)
        self.init_weights = init_weights
        self.init_gate_zero = init_gate_zero
        self.gate_indices = gate_indices
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "paddle":
            pass
        elif self.init_weights == "xavier_uniform":
            init_xavier_uniform_merged_linear(
                self.modulation, num_layers=self.num_outputs
            )
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            init_truncnormal_zero_bias(self.modulation)
        elif self.init_weights == "zero":
            init_zero(self.modulation)
        else:
            raise NotImplementedError
        if self.init_gate_zero:
            assert self.gate_indices is not None
            for gate_index in self.gate_indices:
                start = self.out_dim * gate_index
                end = self.out_dim * (gate_index + 1)
                with paddle.no_grad():
                    self.modulation.weight[start:end] = 0
                    self.modulation.bias[start:end] = 0

    def forward(self, cond):
        return self.modulation(cond).chunk(self.num_outputs, dim=1)
