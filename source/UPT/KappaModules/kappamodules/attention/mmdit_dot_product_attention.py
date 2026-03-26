import paddle.nn.functional as F
import einops
import paddle
from paddle_utils import *

from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_merged_linear,
                               init_xavier_uniform_zero_bias)


class MMDiTDotProductAttention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_modalities=2,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        channel_first=False,
        init_weights="truncnormal002",
        init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_modalities = num_modalities
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.qkv = paddle.nn.ModuleList(
            [
                paddle.compat.nn.Linear(dim, dim * 3, bias=qkv_bias)
                for _ in range(num_modalities)
            ]
        )
        self.proj = paddle.nn.ModuleList(
            [
                paddle.compat.nn.Linear(dim, dim, bias=proj_bias)
                for _ in range(num_modalities)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            for qkv in self.qkv:
                init_xavier_uniform_merged_linear(qkv, num_layers=3)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            for proj in self.proj:
                paddle.nn.init.zeros_(proj.weight)
                if proj.bias is not None:
                    paddle.nn.init.zeros_(proj.bias)

    @staticmethod
    def to_channel_last(x):
        return einops.rearrange(x, "b c l -> b l c")

    @staticmethod
    def to_channel_first(x):
        return einops.rearrange(x, "b l c -> b c l")

    def forward(self, *args, attn_mask=None):
        x = list(args)
        assert len(x) == self.num_modalities
        assert all(x[i].ndim == 3 for i in range(self.num_modalities))
        if self.channel_first:
            for i in range(self.num_modalities):
                x[i] = self.to_channel_last(x[i])
        seqlens = [x[i].size(1) for i in range(self.num_modalities)]
        qs = []
        ks = []
        vs = []
        for i in range(self.num_modalities):
            q, k, v = einops.rearrange(
                self.qkv[i](x[i]),
                "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
                three=3,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            ).unbind(0)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        q = paddle.concat(qs, dim=2)
        k = paddle.concat(ks, dim=2)
        v = paddle.concat(vs, dim=2)
        x = paddle.compat.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        x = einops.rearrange(
            x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)"
        )
        x = list(x.split(seqlens, dim=1))
        for i in range(self.num_modalities):
            x[i] = self.proj[i](x[i])
        if self.channel_first:
            for i in range(self.num_modalities):
                x[i] = self.to_channel_first(x[i])
        return x
