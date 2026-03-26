import einops
from kappamodules.layers import LinearProjection
from kappamodules.unet import UnetGino
from kappautils.param_checking import to_ntuple
from models.base.single_model_base import SingleModelBase


class UnetGinoModel(SingleModelBase):
    """Unet model from GINO"""

    def __init__(self, dim, depth=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        seqlen, input_dim = self.input_shape
        self.output_shape = seqlen, dim
        self.unet = UnetGino(input_dim=input_dim, hidden_dim=dim, depth=depth)

    def forward(self, x, condition=None):
        assert condition is None
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        x = self.unet(x)
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
