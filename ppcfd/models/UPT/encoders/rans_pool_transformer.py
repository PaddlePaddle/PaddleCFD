from functools import partial

import paddle
from kappamodules.layers import LinearProjection
from kappamodules.transformer import (DitPerceiverPoolingBlock,
                                      PerceiverPoolingBlock, PrenormBlock)
from kappamodules.vit import DitBlock
from models.base.single_model_base import SingleModelBase
from modules.gno.rans_pool import RansPool
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import \
    ExcludeFromWdByNameModifier
from torch_geometric.utils import to_dense_batch


class RansPoolTransformer(SingleModelBase):
    def __init__(
        self,
        gnn_dim,
        enc_dim,
        enc_depth,
        enc_num_attn_heads,
        add_type_token=False,
        init_weights="xavier_uniform",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.add_type_token = add_type_token
        self.init_weights = init_weights
        self.mesh_embed = RansPool(
            hidden_dim=gnn_dim, init_weights=init_weights, ndim=self.static_ctx["ndim"]
        )
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        self.blocks = paddle.nn.LayerList(
            [
                PrenormBlock(
                    dim=enc_dim, num_heads=enc_num_attn_heads, init_weights=init_weights
                )
                for _ in range(enc_depth)
            ]
        )
        if add_type_token:
            self.type_token = paddle.nn.Parameter(paddle.empty(size=(1, 1, enc_dim)))
        else:
            self.type_token = None
        self.output_shape = None, enc_dim

    def model_specific_initialization(self):
        if self.add_type_token:
            paddle.nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = []
        if self.add_type_token:
            modifiers += [ExcludeFromWdByNameModifier(name="type_token")]
        return modifiers

    def forward(self, mesh_pos, mesh_edges, batch_idx):
        x = self.mesh_embed(
            mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx
        )
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x)
        if self.add_type_token:
            x = x + self.type_token
        return x
