import einops
from kappamodules.unet import UnetDenoisingDiffusion
from models.base.single_model_base import SingleModelBase


class UnetDenoisingDiffusionModel(SingleModelBase):
    def __init__(self, dim, depth, num_attn_heads=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        seqlen, input_dim = self.input_shape
        self.output_shape = seqlen, dim
        self.model = UnetDenoisingDiffusion(
            dim=dim,
            dim_in=input_dim,
            ndim=self.static_ctx["ndim"],
            num_heads=num_attn_heads,
            depth=depth,
            dim_cond=self.static_ctx.get("condition_dim", None),
        )

    def forward(self, x, condition=None):
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        x = self.model(x, cond=condition)
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
