import einops
import numpy as np
import paddle
from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_interpolate_grid_to_mesh import CfdInterpolateGridToMesh


class CfdInterpolate(SingleModelBase):
    def __init__(self, dim=None, clamp=None, clamp_mode="log", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        _, input_dim = self.input_shape
        _, output_dim = self.output_shape
        self.grid_to_mesh = CfdInterpolateGridToMesh()
        hidden_dim = dim or input_dim
        self.pred = paddle.nn.Sequential(
            paddle.compat.nn.Linear(input_dim, hidden_dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(hidden_dim, output_dim),
        )
        self.resolution = self.static_ctx["grid_resolution"]

    def forward(self, x, grid_pos, query_pos, grid_to_query_edges):
        assert len(query_pos) % len(x) == 0
        query_pos = einops.rearrange(
            query_pos,
            "(batch_size num_query_pos) ndim -> batch_size num_query_pos ndim",
            batch_size=len(x),
        )
        x = x.reshape(len(x), *self.resolution, -1)
        x = einops.rearrange(
            x, "batch_size height width dim -> batch_size dim width height"
        )
        x = self.grid_to_mesh(x, query_pos=query_pos)
        x = self.pred(x)
        if self.clamp is not None:
            assert self.clamp_mode == "log"
            x = paddle.sign(x) * (
                self.clamp + paddle.log(1 + x.abs()) - np.log(1 + self.clamp)
            )
        return x
