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
from paddle_geometric.utils import to_dense_batch


class RansPoolTransformerPerceiver(SingleModelBase):
    def __init__(
        self,
        gnn_dim,
        enc_dim,
        perc_dim,
        enc_depth,
        enc_num_attn_heads,
        perc_num_attn_heads,
        num_latent_tokens,
        use_enc_norm=False,
        add_type_token=False,
        init_weights="xavier_uniform",
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
        self.add_type_token = add_type_token
        self.init_weights = init_weights
        _, input_dim = self.input_shape
        self.static_ctx["ndim"] = input_dim
        self.mesh_embed = RansPool(
            hidden_dim=gnn_dim, init_weights=init_weights, ndim=input_dim
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
                    dim=enc_dim, num_heads=enc_num_attn_heads, init_weights=init_weights
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
        if add_type_token:
            self.type_token = paddle.nn.Parameter(paddle.empty(size=(1, 1, enc_dim)))
        else:
            self.type_token = None
        self.output_shape = num_latent_tokens, perc_dim

    def model_specific_initialization(self):
        if self.add_type_token:
            paddle.nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = [ExcludeFromWdByNameModifier(name="perceiver.query")]
        if self.add_type_token:
            modifiers += [ExcludeFromWdByNameModifier(name="type_token")]
        return modifiers

    def forward(self, mesh_pos, mesh_edges, batch_idx):
        x = self.mesh_embed(
            mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx
        )
        x = self.enc_norm(x)
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.perc_proj(x)
        x = self.perceiver(kv=x)
        if self.add_type_token:
            x = x + self.type_token
        return x
