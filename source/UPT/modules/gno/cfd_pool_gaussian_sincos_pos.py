import einops
import numpy as np
import paddle
from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_zero_bias)
from kappamodules.layers import (ContinuousSincosEmbed, LinearProjection,
                                 Residual)
from paddle_utils import *
from paddle_scatter import segment_csr


class CfdPoolGaussianSincosPos(paddle.nn.Layer):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        positional_std,
        ndim=2,
        init_weights="xavier_uniform",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights
        self.input_proj = paddle.nn.Sequential(
            paddle.compat.nn.Linear(input_dim, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim),
        )
        generator = paddle.Generator().manual_seed(42)
        self.register_buffer(
            "b",
            paddle.normal(mean=paddle.zeros(hidden_dim // 2, ndim), std=positional_std),
        )
        self.message = paddle.nn.Sequential(
            paddle.compat.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim * 2, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, x, mesh_pos, mesh_edges, batch_idx):
        assert x.ndim == 2
        assert mesh_pos.ndim == 2
        assert mesh_edges.ndim == 2
        x = self.input_proj(x) + self.pos_embed(mesh_pos)
        dst_idx, src_idx = mesh_edges.unbind(1)
        x = paddle.concat([x[src_idx], x[dst_idx]], dim=1)
        x = self.message(x)
        dst_indices, counts = dst_idx.unique(return_counts=True)
        padded_counts = paddle.zeros(
            len(counts) + 1, device=counts.device, dtype=counts.dtype
        )
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")
        batch_size = batch_idx._max() + 1
        assert dst_indices.size % batch_size == 0
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )
        return x

    def pos_embed(self, pos):
        return paddle.concat(
            [
                paddle.cos(2.0 * paddle.pi * pos @ self.b.T),
                paddle.sin(2.0 * paddle.pi * pos @ self.b.T),
            ],
            dim=-1,
        )
