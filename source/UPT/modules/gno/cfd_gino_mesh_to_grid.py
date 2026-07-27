import einops
import numpy as np
import paddle
from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_zero_bias)
from kappamodules.layers import (ContinuousSincosEmbed, LinearProjection,
                                 Residual)
from paddle_scatter import segment_csr


class CfdGinoMeshToGrid(paddle.nn.Layer):
    def __init__(
        self, input_dim, hidden_dim, resolution, init_weights="xavier_uniform"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        self.init_weights = init_weights
        self.num_grid_points = int(np.prod(resolution))
        self.input_proj = paddle.nn.Sequential(
            paddle.compat.nn.Linear(input_dim, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=len(resolution))
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

    def forward(self, x, mesh_pos, grid_pos, mesh_to_grid_edges):
        assert x.ndim == 2
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        assert mesh_to_grid_edges.ndim == 2
        assert len(grid_pos) % self.num_grid_points == 0
        x = self.input_proj(x) + self.pos_embed(mesh_pos)
        grid_pos = self.pos_embed(grid_pos)
        grid_idx, mesh_idx = mesh_to_grid_edges.unbind(1)
        x = paddle.concat([x[mesh_idx], grid_pos[grid_idx]], dim=1)
        x = self.message(x)
        dst_indices, counts = grid_idx.unique(return_counts=True)
        padded_counts = paddle.zeros(
            len(grid_pos) + 1, device=counts.device, dtype=counts.dtype
        )
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")
        x = x.reshape(-1, *self.resolution, self.output_dim)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size (...) dim")
        return x
