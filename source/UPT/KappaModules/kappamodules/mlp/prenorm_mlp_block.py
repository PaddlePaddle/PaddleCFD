import paddle

from kappamodules.init.functional import init_norms_as_noaffine
from kappamodules.layers import DropPath, LayerScale

from .mlp import Mlp


class PrenormMlpBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        mlp_hidden_dim=None,
        mlp_bias=True,
        drop_path=0.0,
        norm_ctor=paddle.nn.LayerNorm,
        mlp_ctor=Mlp,
        layerscale=None,
        eps=1e-06,
        init_weights="xavier_uniform",
        init_norms="nonaffine",
        init_last_proj_zero=False,
    ):
        super().__init__()
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        self.norm = norm_ctor(dim, eps=eps)
        self.mlp = mlp_ctor(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
            bias=mlp_bias,
        )
        self.ls = (
            paddle.nn.Identity()
            if layerscale is None
            else LayerScale(dim, init_scale=layerscale)
        )
        self.drop_path = DropPath(drop_prob=drop_path)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_norms == "torch":
            pass
        elif self.init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm)
        else:
            raise NotImplementedError

    def _mlp_residual_path(self, x):
        return self.ls(self.mlp(self.norm(x)))

    def forward(self, x):
        x = self.drop_path(x, self._mlp_residual_path)
        return x
