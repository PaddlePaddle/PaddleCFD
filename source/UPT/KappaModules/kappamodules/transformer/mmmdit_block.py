import paddle

from kappamodules.attention import MMMDiTDotProductAttention
from kappamodules.modulation import Dit
from kappamodules.modulation.functional import (modulate_gate,
                                                modulate_scale_shift)
from kappamodules.utils.param_checking import to_ntuple

from .mlp import Mlp


class MMMDitBlock(paddle.nn.Layer):
    """
    modular multi-modal adaptive norm block
    adaption of the multi-modal adaptive norm block (https://arxiv.org/abs/2403.03206)
    - supports arbitrary many modalities
    - conditioning is assumed to be per modality
    - interaction is split into "main-branches" and "optional-branches"
        - "main-branches" interact with each other and also propagate gradients through
        - "optional-branches" interact with itself (with gradient) and with the "main-branches" (without gradient)
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_main_modalities,
        num_optional_modalities,
        cond_dim=None,
        qkv_bias=True,
        act_ctor=paddle.nn.GELU,
        eps=1e-06,
        init_weights="truncnormal002",
        init_last_proj_zero=False,
        init_gate_zero=False,
    ):
        super().__init__()
        num_modalities = num_main_modalities + num_optional_modalities
        dim = to_ntuple(dim, num_modalities)
        self.num_modalities = num_modalities
        self.dim = dim
        self.num_heads = num_heads
        self.num_main_modalities = num_main_modalities
        self.num_optional_modalities = num_optional_modalities
        self.eps = eps
        self.modulation = paddle.nn.ModuleList(
            [
                Dit(
                    cond_dim=cond_dim,
                    out_dim=dim[i],
                    init_weights=init_weights,
                    num_outputs=6,
                    gate_indices=[2, 5],
                    init_gate_zero=init_gate_zero,
                )
                for i in range(num_modalities)
            ]
        )
        assert all(
            dim[0] == dim[i] for i in range(num_main_modalities)
        ), "main_modalities need same dim"
        self.attn = MMMDiTDotProductAttention(
            dim=dim[0],
            num_heads=num_heads,
            num_main_modalities=num_main_modalities,
            num_optional_modalities=num_optional_modalities,
            qkv_bias=qkv_bias,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.mlp = paddle.nn.ModuleList(
            [
                Mlp(
                    in_dim=dim[i],
                    hidden_dim=dim[i] * 4,
                    act_ctor=act_ctor,
                    init_weights=init_weights,
                    init_last_proj_zero=init_last_proj_zero,
                )
                for i in range(num_modalities)
            ]
        )

    def _attn_residual_path(self, x, scales, shifts, gates, attn_mask):
        assert isinstance(x, list)
        assert isinstance(scales, list)
        assert isinstance(shifts, list)
        assert isinstance(gates, list)
        assert len(x) == len(scales)
        assert len(x) == len(shifts)
        assert len(x) == len(gates)
        for i in range(len(x)):
            x[i] = paddle.nn.functional.layer_norm(x[i], [self.dim[i]], eps=self.eps)
            x[i] = modulate_scale_shift(x[i], scale=scales[i], shift=shifts[i])
        x = self.attn(*x, attn_mask=attn_mask)
        for i in range(len(x)):
            x[i] = modulate_gate(x[i], gate=gates[i])
        return x

    def _mlp_residual_path(self, x, scales, shifts, gates):
        assert isinstance(x, list)
        assert isinstance(scales, list)
        assert isinstance(shifts, list)
        assert isinstance(gates, list)
        assert len(x) == len(scales)
        assert len(x) == len(shifts)
        assert len(x) == len(gates)
        for i in range(len(x)):
            x[i] = paddle.nn.functional.layer_norm(x[i], [self.dim[i]], eps=self.eps)
            x[i] = modulate_scale_shift(x[i], scale=scales[i], shift=shifts[i])
            x[i] = self.mlp[i](x[i])
            x[i] = modulate_gate(x[i], gate=gates[i])
        return x

    def forward(self, x, cond, attn_mask=None):
        assert isinstance(x, (list, tuple))
        assert isinstance(cond, (list, tuple))
        scales_shifts_gates = [
            self.modulation[i](cond[i]) for i in range(self.num_modalities)
        ]
        scales = [scales_shifts_gates[i][0] for i in range(self.num_modalities)]
        shifts = [scales_shifts_gates[i][1] for i in range(self.num_modalities)]
        gates = [scales_shifts_gates[i][2] for i in range(self.num_modalities)]
        og_x = [x[i] for i in range(len(x))]
        x = self._attn_residual_path(
            x=x, scales=scales, shifts=shifts, gates=gates, attn_mask=attn_mask
        )
        x = [(og_x[i] + x[i]) for i in range(len(x))]
        scales = [scales_shifts_gates[i][3] for i in range(self.num_modalities)]
        shifts = [scales_shifts_gates[i][4] for i in range(self.num_modalities)]
        gates = [scales_shifts_gates[i][5] for i in range(self.num_modalities)]
        og_x = [x[i] for i in range(len(x))]
        x = self._mlp_residual_path(x=x, scales=scales, shifts=shifts, gates=gates)
        x = [(og_x[i] + x[i]) for i in range(len(x))]
        return x
