import paddle
import paddle.nn.functional as F
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
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rel_pos_bias = rel_pos_bias
        self.seqlens = seqlens
        self.channel_first = channel_first
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.qkv = paddle.nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
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
        self.proj = paddle.nn.Linear(dim, dim, bias_attr=proj_bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "paddle":
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
        qkv = paddle.reshape(
            self.qkv(x),
            shape=[0, 0, 3, self.num_heads, self.head_dim],
        )
        q, k, v = paddle.unstack(qkv, axis=2)
        if self.rel_pos_bias_table is not None:
            assert attn_mask is None
            seqlen = x.shape[1]
            assert self.rel_pos_idx.shape == (
                seqlen,
                seqlen,
            ), f"invalid input seqlen {seqlen} (expected {self.rel_pos_idx.shape[0]})"
            rel_pos_idx = paddle.reshape(self.rel_pos_idx, shape=[-1])
            attn_mask = paddle.reshape(
                self.rel_pos_bias_table[rel_pos_idx],
                shape=[*self.rel_pos_idx.shape, -1],
            )
            attn_mask = paddle.transpose(attn_mask, perm=[2, 0, 1]).unsqueeze(0)
            attn_mask = attn_mask.cast(q.dtype)

        orig_dtype = x.dtype
        if x.dtype == paddle.float32:
            # 使用 bfloat16 是最推荐的，因为它动态范围大，CFD 模拟不容易崩
            q = q.cast(paddle.bfloat16)
            k = k.cast(paddle.bfloat16)
            v = v.cast(paddle.bfloat16)
            if attn_mask is not None:
                attn_mask = attn_mask.cast(paddle.bfloat16)
        x = paddle.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        if x.dtype != orig_dtype:
            x = x.cast(orig_dtype)
        x = paddle.flatten(x, start_axis=2, stop_axis=3)


        # scale = 1.0 / paddle.sqrt(paddle.to_tensor(self.head_dim, dtype=q.dtype))
        # attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale
        # if attn_mask is not None:
        #     attn = attn + attn_mask
        # attn = paddle.nn.functional.softmax(attn, axis=-1)
        # x = paddle.matmul(attn, v)
        # x = einops.rearrange(
        #     x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)"
        # )
        x = self.proj(x)
        if self.channel_first:
            x = self.to_channel_first(x, og_shape=og_shape)
        return x


class DotProductAttention1d(DotProductAttention):
    def to_channel_last(self, x):
        return paddle.transpose(x, perm=[0, 2, 1])

    def to_channel_first(self, x, og_shape):
        return paddle.transpose(x, perm=[0, 2, 1])


class DotProductAttention2d(DotProductAttention):
    def to_channel_last(self, x):
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        return paddle.flatten(x, start_axis=1, stop_axis=2)

    def to_channel_first(self, x, og_shape):
        _, _, h, w = og_shape
        x = paddle.reshape(x, shape=[x.shape[0], h, w, x.shape[-1]])
        return paddle.transpose(x, perm=[0, 3, 1, 2])


class DotProductAttention3d(DotProductAttention):
    def to_channel_last(self, x):
        x = paddle.transpose(x, perm=[0, 2, 3, 4, 1])
        return paddle.flatten(x, start_axis=1, stop_axis=3)

    def to_channel_first(self, x, og_shape):
        _, _, h, w, d = og_shape
        x = paddle.reshape(x, shape=[x.shape[0], h, w, d, x.shape[-1]])
        return paddle.transpose(x, perm=[0, 4, 1, 2, 3])
