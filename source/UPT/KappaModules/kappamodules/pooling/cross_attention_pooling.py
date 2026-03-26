import einops
import paddle

from kappamodules.layers import LinearProjection


class CrossAttentionPooling(paddle.nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        out_dim: int = None,
        kv_bias: bool = True,
        num_query_tokens: int = 1,
        init_std: float = 0.02,
        normalize_q=False,
        normalize_kv=True,
        init_weights: str = "truncnormal002",
        eps=1e-06,
    ):
        super().__init__()
        assert hasattr(paddle.nn.functional, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        out_dim = out_dim or dim
        self.out_dim = out_dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.init_std = init_std
        self.normalize_q = normalize_q
        self.normalize_kv = normalize_kv
        self.eps = eps
        if normalize_kv:
            self.norm = paddle.nn.LayerNorm(dim, eps=eps)
        else:
            self.norm = paddle.nn.Identity()
        if num_query_tokens > 0:
            self.query_tokens = paddle.nn.Parameter(
                paddle.zeros(1, num_query_tokens, dim)
            )
        else:
            self.query_tokens = None
        if normalize_q:
            self.norm_q = paddle.nn.LayerNorm(dim, eps=eps)
        else:
            self.norm_q = paddle.nn.Identity()
        self.kv = LinearProjection(
            dim, dim * 2, bias=kv_bias, init_weights=init_weights
        )
        self.out = LinearProjection(
            dim, out_dim, init_weights=init_weights, optional=True
        )
        self.num_query_tokens = num_query_tokens
        self.reset_parameters()

    def reset_parameters(self):
        if self.query_tokens is not None:
            paddle.nn.init.normal_(self.query_tokens, std=self.init_std)
        self.kv.reset_parameters()

    def forward(self, x, query_tokens=None):
        if self.query_tokens is None:
            assert (
                query_tokens is not None
            ), f"num_query_tokens == 0 -> requires external query tokens"
            assert query_tokens.ndim == 3
            assert len(query_tokens) == len(x)
            assert query_tokens.size(2) == self.dim
        else:
            assert query_tokens is None, f"attention pooling learns own query tokens"
        if query_tokens is None:
            query_tokens = self.query_tokens.expand(len(x), -1, -1)
        query_tokens = self.norm_q(query_tokens)
        q = einops.rearrange(
            query_tokens,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        kv = self.kv(self.norm(x))
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        x = paddle.compat.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(
            x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)"
        )
        x = self.out(x)
        return x
