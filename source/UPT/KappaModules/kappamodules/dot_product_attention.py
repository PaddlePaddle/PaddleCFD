import einops
import paddle

from kappamodules.functional.pos_embed import relative_position_indices
from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_merged_linear,
                               init_xavier_uniform_zero_bias)


class DotProductAttention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        rel_pos_bias="none",
        seqlens=None,
        channel_first=False,
        init_weights="truncnormal002",
        init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(paddle.nn.functional, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rel_pos_bias = rel_pos_bias
        self.seqlens = seqlens
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.qkv = paddle.compat.nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias == "none":
            self.rel_pos_bias_table = None
            self.rel_pos_idx = None
        elif rel_pos_bias == "learnable":
            assert seqlens is not None
            rel_pos_idx, num_distinct_distances = relative_position_indices(
                seqlens=seqlens, num_aux_tokens=1
            )
            self.register_buffer("rel_pos_idx", rel_pos_idx)
            self.rel_pos_bias_table = paddle.nn.Parameter(
                paddle.empty(num_distinct_distances, num_heads)
            )
        else:
            raise NotImplementedError
        self.proj = paddle.compat.nn.Linear(dim, dim, bias=proj_bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.qkv, num_layers=3)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.rel_pos_bias_table is not None:
            paddle.nn.init.zeros_(self.rel_pos_bias_table)
        if self.init_last_proj_zero:
            paddle.nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                paddle.nn.init.zeros_(self.proj.bias)

    def to_channel_last(self, x):
        raise NotImplementedError

    def to_channel_first(self, x, og_shape):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        if self.channel_first:
            og_shape = x.shape
            x = self.to_channel_last(x)
        else:
            og_shape = None
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        if self.rel_pos_bias_table is not None:
            assert attn_mask is None
            seqlen = x.size(1)
            assert self.rel_pos_idx.shape == (
                seqlen,
                seqlen,
            ), f"invalid input seqlen {seqlen} (expected {self.rel_pos_idx.shape[0]})"
            attn_mask = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)].view(
                *self.rel_pos_idx.shape, -1
            )
            attn_mask = (
                einops.rearrange(attn_mask, "... num_heads -> 1 num_heads ...")
                .contiguous()
                .to(q.dtype)
            )
        x = paddle.compat.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        x = einops.rearrange(
            x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)"
        )
        x = self.proj(x)
        if self.channel_first:
            x = self.to_channel_first(x, og_shape=og_shape)
        return x


class DotProductAttention1d(DotProductAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c l -> b l c")

    def to_channel_first(self, x, og_shape):
        return einops.rearrange(x, "b l c -> b c l")


class DotProductAttention2d(DotProductAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c h w -> b (h w) c")

    def to_channel_first(self, x, og_shape):
        _, _, h, w = og_shape
        return einops.rearrange(x, "bs (h w) dim -> bs dim h w", h=h, w=w)


class DotProductAttention3d(DotProductAttention):
    def to_channel_last(self, x):
        return einops.rearrange(x, "b c h w d -> b (h w d) c")

    def to_channel_first(self, x, og_shape):
        _, _, h, w, d = og_shape
        return einops.rearrange(x, "bs (h w d) dim -> bs dim h w d", h=h, w=w, d=d)
