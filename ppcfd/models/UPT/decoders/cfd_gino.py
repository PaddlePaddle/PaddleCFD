import numpy as np
import paddle
from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_gino_grid_to_mesh import CfdGinoGridToMesh


class CfdGino(SingleModelBase):
    def __init__(self, hidden_dim, clamp=None, clamp_mode="log", **kwargs):
        super().__init__(**kwargs)
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        _, input_dim = self.input_shape
        _, output_dim = self.output_shape
        self.grid_to_mesh = CfdGinoGridToMesh(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            ndim=self.static_ctx["ndim"],
        )

    def forward(self, x, grid_pos, query_pos, grid_to_query_edges):
        x = self.grid_to_mesh(
            x, query_pos=query_pos, grid_to_query_edges=grid_to_query_edges
        )
        if self.clamp is not None:
            assert self.clamp_mode == "log"
            x = paddle.sign(x) * (
                self.clamp + paddle.log(1 + x.abs()) - np.log(1 + self.clamp)
            )
        return x
