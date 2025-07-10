# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Callable, Optional

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
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb)
        x = x.astype(emb.dtype)
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


def default(val: Optional[Any], d: Callable[[], Any] | Any) -> Any:
    return val if val is not None else (d() if callable(d) else d)


def get_normalization_layer(name, dims, num_groups=None, *args, **kwargs):
    if not isinstance(name, str) or name.lower() == "none":
        return None
    elif "batch_norm" == name:
        return paddle.nn.BatchNorm2D(num_features=dims, *args, **kwargs)
    elif "layer_norm" == name:
        return paddle.nn.LayerNorm(dims, *args, **kwargs)
    elif "instance" in name:
        return paddle.nn.InstanceNorm1D(num_features=dims, *args, **kwargs)
    elif "group" in name:
        if num_groups is None:
            pos_groups = [int(dims / N) for N in range(2, 17) if dims % N == 0]
            if len(pos_groups) == 0:
                raise NotImplementedError(f"Group norm could not infer the number of groups for dim={dims}")
            num_groups = max(pos_groups)
        return paddle.nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)
