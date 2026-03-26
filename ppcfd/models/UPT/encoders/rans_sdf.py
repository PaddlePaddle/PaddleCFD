import numpy as np
import paddle
from kappamodules.layers import ContinuousSincosEmbed
from kappamodules.layers.continuous_sincos_embed import ContinuousSincosEmbed
from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_mesh_to_grid_sdf import RansGinoMeshToGridSdf


class RansSdf(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.sdf_embed = paddle.nn.Sequential(
            paddle.compat.nn.Linear(1, dim),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(dim, dim),
        )
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=len(self.resolution))
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = int(np.prod(self.resolution)), dim

    def forward(self, sdf, grid_pos):
        assert sdf.size(-1) == 1
        sdf_embed = self.sdf_embed(sdf.view(-1, 1))
        grid_pos_embed = self.pos_embed(grid_pos)
        embed = sdf_embed + grid_pos_embed
        embed = embed.view(len(sdf), *self.resolution, -1)
        return embed
