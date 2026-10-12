import paddle

from kappamodules.attention import MMDiTDotProductAttention
from kappamodules.modulation import Dit
from kappamodules.modulation.functional import (modulate_gate,
                                                modulate_scale_shift)

from .mlp import Mlp


class MMDitBlock(paddle.nn.Layer):
    """
    adaption of the multi-modal adaptive norm block (https://arxiv.org/abs/2403.03206)
    - supports arbitrary many modalities
    - conditioning is assumed to be per modality
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_hidden_dim=None,
        num_modalities=2,
        cond_dim=None,
        qkv_bias=True,
        act_ctor=paddle.nn.GELU,
        eps=1e-06,
        init_weights="truncnormal002",
        init_last_proj_zero=False,
        init_gate_zero=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.eps = eps
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        cond_dim = cond_dim or dim
        self.modulation = paddle.nn.ModuleList(
            [
                Dit(
                    cond_dim=cond_dim,
                    out_dim=dim,
                    init_weights=init_weights,
                    num_outputs=6,
                    gate_indices=[2, 5],
                    init_gate_zero=init_gate_zero,
                )
                for _ in range(num_modalities)
            ]
        )
        self.attn = MMDiTDotProductAttention(
            dim=dim,
            num_heads=num_heads,
            num_modalities=num_modalities,
            qkv_bias=qkv_bias,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.mlp = paddle.nn.ModuleList(
            [
                Mlp(
                    in_dim=dim,
                    hidden_dim=mlp_hidden_dim,
                    act_ctor=act_ctor,
                    init_weights=init_weights,
                    init_last_proj_zero=init_last_proj_zero,
                )
                for _ in range(num_modalities)
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
            x[i] = paddle.nn.functional.layer_norm(x[i], [self.dim], eps=self.eps)
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
            x[i] = paddle.nn.functional.layer_norm(x[i], [self.dim], eps=self.eps)
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
