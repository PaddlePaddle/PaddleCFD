import paddle
from einops import rearrange


class LinearAttention(paddle.nn.Layer):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        dropout: float = 0.0,
        rescale: str = "qk",
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = paddle.nn.Sequential(
            paddle.nn.Dropout(p=dropout),
            paddle.nn.Conv2D(
                in_channels=dim,
                out_channels=hidden_dim * 3,
                kernel_size=1,
                bias_attr=False,
            ),
        )
        assert rescale in ["qk", "qkv"]
        self.rescale = getattr(self, f"rescale_{rescale}")
        self.to_out = paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1)
        # nn.Sequential(
        #     nn.Conv2d(hidden_dim, dim, 1),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x):
        b, c, h, w = tuple(x.shape)
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q, k, v = self.rescale(q, k, v, h=h, w=w)
        context = paddle.einsum("b h d n, b h e n -> b h d e", k, v)
        out = paddle.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

    def rescale_qk(self, q, k, v, h, w):
        q = q * self.scale
        k = paddle.nn.functional.softmax(k, axis=-1)
        return q, k, v

    def rescale_qkv(self, q, k, v, h, w):
        q = paddle.nn.functional.softmax(q, axis=-2)
        q = q * self.scale
        k = paddle.nn.functional.softmax(k, axis=-1)
        v = v / (h * w)
        return q, k, v


def l2norm(t):
    return paddle.nn.functional.normalize(x=t, axis=-1)


class Attention(paddle.nn.Layer):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = paddle.nn.Conv2D(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias_attr=False)
        self.to_out = paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1)
        self.dropout = paddle.nn.Dropout(p=dropout)

    def forward(self, x):
        b, c, h, w = tuple(x.shape)
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale
        sim = paddle.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = paddle.nn.functional.softmax(sim, axis=-1)
        attn = self.dropout(attn)
        out = paddle.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
