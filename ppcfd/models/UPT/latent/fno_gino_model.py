import einops
from kappamodules.layers import LinearProjection
from kappautils.param_checking import to_ntuple
from models.base.single_model_base import SingleModelBase
from neuralop.models import FNO


class FnoGinoModel(SingleModelBase):
    """FNO model from GINO"""

    def __init__(
        self,
        modes=32,
        dim=86,
        norm="group_norm",
        factorization="tucker",
        rank=0.4,
        domain_padding=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.modes = to_ntuple(modes, n=self.static_ctx["ndim"])
        self.dim = dim
        self.norm = norm
        self.factorization = factorization
        self.rank = rank
        self.domain_padding = domain_padding
        seqlen, input_dim = self.input_shape
        self.output_shape = seqlen, dim
        self.proj_in = LinearProjection(input_dim, dim)
        self.fno = FNO(
            self.modes,
            in_channels=dim,
            hidden_channels=dim,
            out_channels=dim,
            use_mlp=True,
            mlp_expansion=1.0,
            factorization=factorization,
            domain_padding=domain_padding,
            norm=norm,
            rank=rank,
        )

    def forward(self, x, condition=None):
        assert condition is None
        x = self.proj_in(x)
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        x = self.fno(x)
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
