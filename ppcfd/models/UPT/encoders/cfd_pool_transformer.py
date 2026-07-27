from functools import partial

import paddle
from kappamodules.layers import LinearProjection
from kappamodules.transformer import (DitPerceiverPoolingBlock,
                                      PerceiverPoolingBlock, PrenormBlock)
from kappamodules.vit import DitBlock
from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_pool import CfdPool
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import \
    ExcludeFromWdByNameModifier
from paddle_geometric.utils import to_dense_batch


class CfdPoolTransformer(SingleModelBase):
    def __init__(
        self,
        gnn_dim,
        enc_dim,
        enc_depth,
        enc_num_attn_heads,
        init_weights="xavier_uniform",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.init_weights = init_weights
        _, input_dim = self.input_shape
        self.mesh_embed = CfdPool(
            input_dim=input_dim, hidden_dim=gnn_dim, init_weights=init_weights
        )
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        self.blocks = paddle.nn.LayerList(
            [
                block_ctor(
                    dim=enc_dim, num_heads=enc_num_attn_heads, init_weights=init_weights
                )
                for _ in range(enc_depth)
            ]
        )
        self.output_shape = None, enc_dim

    def forward(
        self, x, mesh_pos, mesh_edges, batch_idx, condition=None, static_tokens=None
    ):
        x = self.mesh_embed(
            x, mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx
        )
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)
        return x
