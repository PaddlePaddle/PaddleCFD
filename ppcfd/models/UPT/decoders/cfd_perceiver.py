from functools import partial

import einops
import paddle
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import DitPerceiverBlock, PerceiverBlock
from models.base.single_model_base import SingleModelBase
from paddle_geometric.utils import unbatch


class CfdPerceiver(SingleModelBase):
    def __init__(self, dim, num_attn_heads, init_weights="xavier_uniform", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.init_weights = init_weights
        seqlen, input_dim = self.input_shape
        self.proj = LinearProjection(input_dim, dim)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=2)
        self.query_mlp = paddle.nn.Sequential(
            LinearProjection(dim, dim * 4),
            paddle.nn.GELU(),
            LinearProjection(dim * 4, dim * 4),
            paddle.nn.GELU(),
            LinearProjection(dim * 4, dim),
        )
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(
                DitPerceiverBlock, cond_dim=self.static_ctx["condition_dim"]
            )
        else:
            block_ctor = PerceiverBlock
        self.perceiver = block_ctor(
            dim=dim, num_heads=num_attn_heads, init_weights=init_weights
        )
        _, num_channels = self.output_shape
        self.pred = LinearProjection(dim, num_channels)

    def forward(
        self,
        x,
        query_pos,
        unbatch_idx,
        unbatch_select,
        static_tokens=None,
        condition=None,
    ):
        assert x.ndim == 3
        x = self.proj(x)
        pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(pos_embed)
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.perceiver(q=query, kv=x, **block_kwargs)
        x = self.pred(x)
        x = einops.rearrange(
            x, "batch_size max_num_points dim -> (batch_size max_num_points) dim"
        )
        unbatched = unbatch(x, batch=unbatch_idx)
        x = paddle.concat([unbatched[i] for i in unbatch_select])
        return x
