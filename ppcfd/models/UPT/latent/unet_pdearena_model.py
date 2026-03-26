import einops
from kappamodules.unet import UnetPdearena
from models.base.single_model_base import SingleModelBase


class UnetPdearenaModel(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        seqlen, input_dim = self.input_shape
        self.output_shape = seqlen, dim
        assert self.static_ctx["ndim"] == 2
        self.model = UnetPdearena(
            hidden_channels=dim,
            input_dim=input_dim,
            output_dim=dim,
            norm=False,
            cond_dim=self.static_ctx.get("condition_dim", None),
        )

    def forward(self, x, condition=None):
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        x = self.model(x, emb=condition)
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
