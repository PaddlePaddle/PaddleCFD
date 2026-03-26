from functools import partial

import paddle
from kappamodules.layers import LinearProjection
from kappamodules.transformer import (DitBlock, DitPerceiverPoolingBlock,
                                      PerceiverPoolingBlock, PrenormBlock)
from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_pool import CfdPool
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import \
    ExcludeFromWdByNameModifier
from torch_geometric.utils import to_dense_batch


class CfdPoolTransformerPerceiver(SingleModelBase):
    def __init__(
        self,
        gnn_dim,
        enc_dim,
        perc_dim,
        enc_depth,
        enc_num_attn_heads,
        perc_num_attn_heads,
        num_latent_tokens=None,
        use_enc_norm=False,
        drop_path_rate=0.0,
        init_weights="xavier_uniform",
        gnn_init_weights=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.perc_dim = perc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.perc_num_attn_heads = perc_num_attn_heads
        self.num_latent_tokens = num_latent_tokens
        self.use_enc_norm = use_enc_norm
        self.drop_path_rate = drop_path_rate
        self.init_weights = init_weights
        gnn_init_weights = gnn_init_weights or init_weights
        self.gnn_init_weights = gnn_init_weights
        _, input_dim = self.input_shape
        self.mesh_embed = CfdPool(
            input_dim=input_dim, hidden_dim=gnn_dim, init_weights=gnn_init_weights
        )
        self.enc_norm = (
            paddle.nn.LayerNorm(gnn_dim, eps=1e-06)
            if use_enc_norm
            else paddle.nn.Identity()
        )
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        self.blocks = paddle.nn.LayerList(
            [
                block_ctor(
                    dim=enc_dim,
                    num_heads=enc_num_attn_heads,
                    init_weights=init_weights,
                    drop_path=drop_path_rate,
                )
                for _ in range(enc_depth)
            ]
        )
        self.perc_proj = LinearProjection(enc_dim, perc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(
                DitPerceiverPoolingBlock,
                perceiver_kwargs=dict(
                    cond_dim=self.static_ctx["condition_dim"], init_weights=init_weights
                ),
            )
        else:
            block_ctor = partial(
                PerceiverPoolingBlock, perceiver_kwargs=dict(init_weights=init_weights)
            )
        self.perceiver = block_ctor(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            num_query_tokens=num_latent_tokens,
        )
        self.output_shape = num_latent_tokens, perc_dim

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="perceiver.query")]

    def forward(
        self, x, mesh_pos, mesh_edges, batch_idx, condition=None, static_tokens=None
    ):
        x = self.mesh_embed(
            x, mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx
        )
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.enc_norm(x)
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)
        x = self.perc_proj(x)
        x = self.perceiver(kv=x, **block_kwargs)
        return x
