import einops
import paddle

from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock


class PerceiverDecoder(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        ndim,
        input_dim,
        output_dim,
        init_weights="truncnormal002",
        eps=1e-06,
    ):
        super().__init__()
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.query = paddle.nn.Sequential(
            LinearProjection(dim, dim, init_weights=init_weights),
            paddle.nn.GELU(),
            LinearProjection(dim, dim, init_weights=init_weights),
        )
        self.proj = LinearProjection(
            input_dim, dim, init_weights=init_weights, optional=True
        )
        self.perc = PerceiverBlock(
            dim=dim, num_heads=num_heads, init_weights=init_weights, eps=eps
        )
        self.norm = paddle.nn.LayerNorm(dim, eps=eps)
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

    def forward(self, x, pos, block_kwargs=None, unbatch_mask=None):
        if pos is None:
            assert (
                not self.training
            ), f"{type(self).__name__} expects query positions during training"
            return None
        assert x.ndim == 3
        assert pos.ndim == 3
        query = self.query(self.pos_embed(pos))
        x = self.proj(x)
        x = self.perc(q=query, kv=x, **block_kwargs or {})
        x = self.norm(x)
        x = self.pred(x)
        x = einops.rearrange(
            x, "batch_size max_num_points dim -> (batch_size max_num_points) dim"
        )
        if len(pos) == 1:
            pass
        elif unbatch_mask is not None:
            x = x[unbatch_mask]
        return x
