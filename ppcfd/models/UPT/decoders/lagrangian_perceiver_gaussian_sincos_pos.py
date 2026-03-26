from functools import partial

import einops
import paddle
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import DitPerceiverBlock, PerceiverBlock
from models.base.single_model_base import SingleModelBase
from torch_geometric.utils import unbatch


class LagrangianPerceiverGaussianSincosPos(SingleModelBase):
    def __init__(self, dim, num_attn_heads, positional_std, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        num_channels, _ = self.output_shape
        _, input_dim = self.input_shape
        ndim = self.data_container.get_dataset().metadata["dim"]
        self.proj = LinearProjection(input_dim, dim)
        generator = paddle.Generator().manual_seed(42)
        self.register_buffer(
            "b", paddle.normal(mean=paddle.zeros(dim // 2, ndim), std=positional_std)
        )
        self.query_mlp = paddle.nn.Sequential(
            paddle.compat.nn.Linear(dim, dim * 4),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(dim * 4, dim * 4),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(dim * 4, dim),
        )
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(
                DitPerceiverBlock, cond_dim=self.static_ctx["condition_dim"]
            )
        else:
            block_ctor = PerceiverBlock
        self.perceiver = block_ctor(dim=dim, num_heads=num_attn_heads)
        self.pred = LinearProjection(dim, num_channels)

    def model_specific_initialization(self):
        self.query_mlp.apply(init_xavier_uniform_zero_bias)

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

    def pos_embed(self, pos):
        return paddle.concat(
            [
                paddle.cos(2.0 * paddle.pi * pos @ self.b.T),
                paddle.sin(2.0 * paddle.pi * pos @ self.b.T),
            ],
            dim=-1,
        )
