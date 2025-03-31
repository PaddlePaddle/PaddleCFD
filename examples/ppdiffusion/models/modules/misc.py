import math

import paddle
from einops import rearrange


class Residual(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(paddle.nn.Layer):
    """sinusoidal positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb).astype(x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class LearnedSinusoidalPosEmb(paddle.nn.Layer):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = self.create_parameter(
            shape=half_dim, dtype=paddle.get_default_dtype(), default_initializer=paddle.nn.initializer.Normal()
        )

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = paddle.concat(x=(freqs.sin(), freqs.cos()), axis=-1)
        fouriered = paddle.concat(x=(x, fouriered), axis=-1)
        return fouriered


def get_time_embedder(
    time_dim: int,
    dim: int,
    learned_sinusoidal_cond: bool = False,
    learned_sinusoidal_dim: int = 16,
):
    if learned_sinusoidal_cond:
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1
    else:
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
    time_emb_mlp = paddle.nn.Sequential(
        sinu_pos_emb,
        paddle.nn.Linear(in_features=fourier_dim, out_features=time_dim),
        paddle.nn.GELU(),
        paddle.nn.Linear(in_features=time_dim, out_features=time_dim),
    )
    return time_emb_mlp
