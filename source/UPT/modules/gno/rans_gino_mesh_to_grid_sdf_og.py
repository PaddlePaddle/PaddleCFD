import einops
import numpy as np
import paddle
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from paddle_scatter import segment_csr


class RansGinoMeshToGridSdfOg(paddle.nn.Layer):
    def __init__(self, hidden_dim, output_dim, resolution):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim + 4
        self.resolution = resolution
        self.num_grid_points = int(np.prod(resolution))
        self.sdf_embed = paddle.nn.Sequential(
            paddle.compat.nn.Linear(1, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.mesh_pos_embed = ContinuousSincosEmbed(
            dim=hidden_dim * 4, ndim=len(resolution) + 1
        )
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim * 3, ndim=len(resolution))
        self.message = paddle.nn.Sequential(
            paddle.compat.nn.Linear(hidden_dim * 10, 512),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(512, 256),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(256, output_dim),
        )

    def forward(self, mesh_pos, sdf, grid_pos, mesh_to_grid_edges):
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        assert mesh_to_grid_edges.ndim == 2
        assert len(grid_pos) % self.num_grid_points == 0
        mesh_pos = mesh_pos / 100 - 1
        grid_pos = grid_pos / 100 - 1
        ones = paddle.ones(
            size=(len(mesh_pos),), dtype=mesh_pos.dtype, device=mesh_pos.device
        ).unsqueeze(1)
        mesh_pos = paddle.concat([mesh_pos, ones], dim=1)
        mesh_pos = self.mesh_pos_embed(mesh_pos)
        grid_pos_embed = self.pos_embed(grid_pos)
        sdf = sdf.view(-1, 1)
        sdf_embed = self.sdf_embed(sdf)
        grid_embed = paddle.concat([grid_pos_embed, sdf_embed], dim=1)
        grid_idx, mesh_idx = mesh_to_grid_edges.unbind(1)
        x = paddle.concat([mesh_pos[mesh_idx], grid_embed[grid_idx]], dim=1)
        x = self.message(x)
        dst_indices, counts = grid_idx.unique(return_counts=True)
        padded_counts = paddle.zeros(
            len(grid_embed) + 1, device=counts.device, dtype=counts.dtype
        )
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")
        x = paddle.concat([grid_pos, sdf, x], dim=1)
        x = x.reshape(-1, *self.resolution, self.output_dim)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size (...) dim")
        return x
