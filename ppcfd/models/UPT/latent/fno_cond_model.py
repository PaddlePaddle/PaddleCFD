from functools import partial

import einops
from models.base.single_model_base import SingleModelBase
from modules.pdearena.conditional_twod_resnet import FourierBasicBlock, ResNet


class FnoCondModel(SingleModelBase):
    """FNO model from PDEArena with conditioning"""

    def __init__(self, dim, modes, depth=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.modes = modes
        self.depth = depth
        seqlen, input_dim = self.input_shape
        self.output_shape = seqlen, dim
        self.model = ResNet(
            input_dim=input_dim,
            hidden_dim=dim,
            cond_dim=self.static_ctx["condition_dim"],
            num_blocks=[1] * depth,
            block=partial(FourierBasicBlock, modes1=modes, modes2=modes),
            norm=False,
        )

    def forward(self, x, condition):
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        x = self.model(x, condition)
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
