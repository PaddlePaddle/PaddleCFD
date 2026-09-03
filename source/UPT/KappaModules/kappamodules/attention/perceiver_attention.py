import paddle

from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_merged_linear,
                               init_xavier_uniform_zero_bias)
import paddle.nn.functional as F

class PerceiverAttention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        kv_dim=None,
        num_heads=8,
        bias=True,
        concat_query_to_kv=False,
        init_weights="truncnormal002",
        init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.concat_query_to_kv = concat_query_to_kv
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.kv = paddle.nn.Linear(kv_dim or dim, dim * 2, bias_attr=bias)
        self.q = paddle.nn.Linear(dim, dim, bias_attr=bias)
        self.proj = paddle.nn.Linear(dim, dim, bias_attr=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "paddle":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.kv, num_layers=2)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            paddle.nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                paddle.nn.init.zeros_(self.proj.bias)

    def forward(self, q, kv, attn_mask=None):
        if self.concat_query_to_kv:
            kv = paddle.concat([kv, q], dim=1)
        kv = self.kv(kv)
        q = self.q(q)
        q = paddle.reshape(
            q,
            shape=[0, 0, self.num_heads, self.head_dim],
        )
        kv = paddle.reshape(
            kv,
            shape=[0, 0, 2, self.num_heads, self.head_dim],
        )
        k, v = paddle.unstack(kv, axis=2)

        orig_dtype = q.dtype
        use_bfloat16 = (
            q.dtype == paddle.float32
            and paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
        if use_bfloat16:
            # Prefer bfloat16 on CUDA for speed while keeping CPU path in float32.
            q = q.cast(paddle.bfloat16)
            k = k.cast(paddle.bfloat16)
            v = v.cast(paddle.bfloat16)
            if attn_mask is not None:
                attn_mask = attn_mask.cast(paddle.bfloat16)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
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
        return x


class PerceiverAttention1d(PerceiverAttention):
    pass
