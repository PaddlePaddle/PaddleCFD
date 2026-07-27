import einops
import numpy as np
import paddle
from kappamodules.layers import (ContinuousSincosEmbed, LinearProjection,
                                 Residual)
from paddle_scatter import segment_csr


class CfdGinoMeshToGridOld(paddle.nn.Layer):
    def __init__(self, input_dim, hidden_dim, resolution):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        self.num_grid_points = int(np.prod(resolution))
        if isinstance(hidden_dim, int):
            self.input_proj = paddle.nn.Sequential(
                paddle.compat.nn.Linear(input_dim, hidden_dim),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(hidden_dim, hidden_dim),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(hidden_dim, hidden_dim),
            )
            self.pos_embed = ContinuousSincosEmbed(
                dim=hidden_dim // 2, ndim=len(resolution)
            )
            self.message = paddle.nn.Sequential(
                paddle.compat.nn.Linear(hidden_dim * 2, hidden_dim * 2),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(hidden_dim * 2, hidden_dim * 2),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.output_dim = hidden_dim
        else:
            assert hidden_dim[0] % 4 == 0
            self.input_proj = paddle.nn.Sequential(
                paddle.compat.nn.Linear(input_dim, hidden_dim[0]),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(hidden_dim[0], hidden_dim[0]),
                paddle.nn.GELU(),
                paddle.compat.nn.Linear(hidden_dim[0], hidden_dim[0] // 2),
            )
            self.pos_embed = ContinuousSincosEmbed(
                dim=hidden_dim[0] // 4, ndim=len(resolution)
            )
            layers = []
            for i in range(len(hidden_dim) - 1):
                layers.append(paddle.compat.nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
                if i < len(hidden_dim) - 2:
                    layers.append(paddle.nn.GELU())
            self.message = paddle.nn.Sequential(*layers)
            self.output_dim = hidden_dim[-1]

    def forward(self, x, mesh_pos, grid_pos, mesh_to_grid_edges):
        assert x.ndim == 2
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        assert mesh_to_grid_edges.ndim == 2
        assert len(grid_pos) % self.num_grid_points == 0
        x = paddle.concat([self.input_proj(x), self.pos_embed(mesh_pos)], dim=1)
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
