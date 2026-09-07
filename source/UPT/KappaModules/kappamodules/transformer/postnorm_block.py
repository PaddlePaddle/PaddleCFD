import paddle

from kappamodules.attention import DotProductAttention1d
from kappamodules.init.functional import init_norms_as_noaffine
from kappamodules.layers import DropPath, LayerScale

from .mlp import Mlp


class PostnormBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_hidden_dim=None,
        qkv_bias=True,
        drop_path=0.0,
        act_ctor=paddle.nn.GELU,
        norm_ctor=paddle.nn.LayerNorm,
        attn_ctor=DotProductAttention1d,
        layerscale=None,
        eps=1e-06,
        init_weights="xavier_uniform",
        init_norms="nonaffine",
        init_last_proj_zero=False,
    ):
        super().__init__()
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        self.attn = attn_ctor(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.ls1 = (
            paddle.nn.Identity()
            if layerscale is None
            else LayerScale(dim, init_scale=layerscale)
        )
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.norm1 = norm_ctor(dim, eps=eps)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            act_ctor=act_ctor,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.ls2 = (
            paddle.nn.Identity()
            if layerscale is None
            else LayerScale(dim, init_scale=layerscale)
        )
        self.drop_path2 = DropPath(drop_prob=drop_path)
        self.norm2 = norm_ctor(dim, eps=eps)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_norms == "paddle":
            pass
        elif self.init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm1)
            init_norms_as_noaffine(self.norm2)
        else:
            raise NotImplementedError

    def _attn_residual_path(self, x):
        return self.ls1(self.attn(x))

    def _mlp_residual_path(self, x):
        return self.ls2(self.mlp(x))

    def forward(self, x):
        x = self.norm1(self.drop_path1(x, self._attn_residual_path))
        x = self.norm2(self.drop_path2(x, self._mlp_residual_path))
        return x
