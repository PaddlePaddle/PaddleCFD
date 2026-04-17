"""
This file contains scOT.

A lot of this file is taken from the transformers library and changed to our purposes. Huggingface Transformers is licensed under
Apache 2.0 License, see trainer.py for details.

We follow https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py
and https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129

The class ConvNeXtBlock is taken from the facebookresearch/ConvNeXt repository and is licensed under the MIT License.
"""

import collections
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paddle


def meshgrid(tensors, indexing="ij"):
    return paddle.meshgrid(*tensors)


def drop_path(
    input_tensor: paddle.Tensor, drop_prob: float = 0.0, training: bool = False
) -> paddle.Tensor:
    if drop_prob == 0.0 or not training:
        return input_tensor
    keep_prob = 1 - drop_prob
    shape = (input_tensor.shape[0],) + (1,) * (input_tensor.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=input_tensor.dtype)
    random_tensor = paddle.floor(random_tensor)
    return input_tensor / keep_prob * random_tensor


class Swinv2DropPath(paddle.nn.Layer):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)


def window_partition(input_feature: paddle.Tensor, window_size: int) -> paddle.Tensor:
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.reshape(
        [
            batch_size,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            num_channels,
        ]
    )
    windows = input_feature.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, num_channels]
    )
    return windows


def window_reverse(
    windows: paddle.Tensor, window_size: int, height: int, width: int
) -> paddle.Tensor:
    num_channels = windows.shape[-1]
    windows = windows.reshape(
        [
            -1,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            num_channels,
        ]
    )
    windows = windows.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, height, width, num_channels]
    )
    return windows


@dataclass
class Swinv2EncoderOutput:
    last_hidden_state: paddle.Tensor
    hidden_states: Optional[Tuple[paddle.Tensor, ...]] = None
    attentions: Optional[Tuple[paddle.Tensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[paddle.Tensor, ...]] = None


@dataclass
class ScOTOutput:
    loss: Optional[paddle.Tensor] = None
    output: Optional[paddle.Tensor] = None
    hidden_states: Optional[Tuple[paddle.Tensor, ...]] = None
    attentions: Optional[Tuple[paddle.Tensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[paddle.Tensor, ...]] = None


class ScOTConfig:
    model_type = "swinv2"
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        num_out_channels=1,
        embed_dim=96,
        depths=None,
        num_heads=None,
        skip_connections=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=None,
        residual_model="convnext",
        use_conditioning=False,
        learn_residual=False,
        chunk_size_feed_forward=0,
        use_return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        pretrained_window_sizes=(0, 0, 0, 0),
        **kwargs,
    ):
        depths = depths if depths is not None else [2, 2, 6, 2]
        num_heads = num_heads if num_heads is not None else [3, 6, 12, 24]
        skip_connections = (
            skip_connections if skip_connections is not None else [True, True, True]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.num_layers = len(self.depths)
        self.num_heads = list(num_heads)
        self.skip_connections = list(skip_connections)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.use_conditioning = use_conditioning
        self.learn_residual = learn_residual if self.use_conditioning else False
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.hidden_size = int(embed_dim * 2 ** (len(self.depths) - 1))
        self.pretrained_window_sizes = tuple(pretrained_window_sizes)
        self.p = p
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls.from_dict(config)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


class LayerNorm(paddle.nn.LayerNorm):
    def forward(self, x, time):
        return super().forward(x)


class ConditionalLayerNorm(paddle.nn.Layer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = paddle.nn.Linear(1, dim)
        self.bias = paddle.nn.Linear(1, dim)

    def forward(self, x, time):
        mean = x.mean(axis=-1, keepdim=True)
        var = (x**2).mean(axis=-1, keepdim=True) - mean**2
        x = (x - mean) / paddle.sqrt(var + self.eps)
        time = time.reshape([-1, 1]).astype(x.dtype)
        weight = self.weight(time).unsqueeze(1)
        bias = self.bias(time).unsqueeze(1)
        if x.ndim == 4:
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return weight * x + bias


class Swinv2SelfAttention(paddle.nn.Layer):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )
        self.pretrained_window_size = pretrained_window_size
        self.logit_scale = self.create_parameter(
            shape=[num_heads, 1, 1],
            default_initializer=paddle.nn.initializer.Assign(
                paddle.log(10 * paddle.ones([num_heads, 1, 1], dtype="float32"))
            ),
        )
        self.continuous_position_bias_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(2, 512, bias_attr=True),
            paddle.nn.ReLU(),
            paddle.nn.Linear(512, num_heads, bias_attr=False),
        )

        relative_coords_h = paddle.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype="float32"
        )
        relative_coords_w = paddle.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype="float32"
        )
        relative_coords_table = paddle.stack(
            meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
        )
        relative_coords_table = (
            relative_coords_table.transpose([1, 2, 0]).unsqueeze(0).astype("float32")
        )
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        elif window_size > 1:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = paddle.sign(relative_coords_table) * paddle.log2(
            paddle.abs(relative_coords_table) + 1.0
        ) / math.log2(8)
        relative_coords_table = relative_coords_table.astype(
            self.continuous_position_bias_mlp[0].weight.dtype
        )
        self.register_buffer(
            "relative_coords_table", relative_coords_table, persistable=False
        )

        coords_h = paddle.arange(self.window_size[0], dtype="int64")
        coords_w = paddle.arange(self.window_size[1], dtype="int64")
        coords = paddle.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(axis=-1)
        self.register_buffer(
            "relative_position_index", relative_position_index, persistable=False
        )

        self.query = paddle.nn.Linear(self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias)
        self.key = paddle.nn.Linear(self.all_head_size, self.all_head_size, bias_attr=False)
        self.value = paddle.nn.Linear(self.all_head_size, self.all_head_size, bias_attr=config.qkv_bias)
        self.dropout = paddle.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, ...]:
        batch_size, dim, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Add a tiny constant before normalize to prevent NaN in Paddle's
        # backward when input contains all-zero tokens (padding regions).
        # Paddle's normalize backward divides by the L2 norm, and for zero
        # inputs this produces extremely large gradients that overflow float32.
        # This has no meaningful effect on non-zero tokens.
        _NORM_STABILITY_EPS = 1e-6
        attention_scores = paddle.matmul(
            paddle.nn.functional.normalize(query_layer + _NORM_STABILITY_EPS, axis=-1),
            paddle.nn.functional.normalize(key_layer + _NORM_STABILITY_EPS, axis=-1).transpose([0, 1, 3, 2]),
        )
        logit_scale = paddle.exp(
            paddle.clip(self.logit_scale, max=math.log(1.0 / 0.01))
        )
        attention_scores = attention_scores * logit_scale
        relative_position_bias_table = self.continuous_position_bias_mlp(
            self.relative_coords_table
        ).reshape([-1, self.num_attention_heads])
        relative_position_bias = paddle.index_select(
            relative_position_bias_table,
            self.relative_position_index.reshape([-1]),
            axis=0,
        ).reshape(
            [
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ]
        )
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        relative_position_bias = 16 * paddle.nn.functional.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.reshape(
                [batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim]
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.reshape(
                [-1, self.num_attention_heads, dim, dim]
            )

        attention_probs = paddle.nn.functional.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_shape = list(context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class Swinv2SelfOutput(paddle.nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = paddle.nn.Linear(dim, dim)
        self.dropout = paddle.nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Swinv2Attention(paddle.nn.Layer):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        self.self = Swinv2SelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=(
                pretrained_window_size
                if isinstance(pretrained_window_size, collections.abc.Iterable)
                else (pretrained_window_size, pretrained_window_size)
            ),
        )
        self.output = Swinv2SelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) > 0:
            raise NotImplementedError("Head pruning is not supported in Paddle ScOT.")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, ...]:
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class Swinv2Intermediate(paddle.nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = paddle.nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            act_map = {
                "gelu": paddle.nn.functional.gelu,
                "relu": paddle.nn.functional.relu,
            }
            self.intermediate_act_fn = act_map[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Swinv2Output(paddle.nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = paddle.nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = paddle.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ConvNeXtBlock(paddle.nn.Layer):
    def __init__(self, config, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = paddle.nn.Conv2D(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        layer_norm = ConditionalLayerNorm if config.use_conditioning else LayerNorm
        self.norm = layer_norm(dim, eps=config.layer_norm_eps)
        self.pwconv1 = paddle.nn.Linear(dim, 4 * dim)
        self.act = paddle.nn.GELU()
        self.pwconv2 = paddle.nn.Linear(4 * dim, dim)
        self.weight = (
            self.create_parameter(
                shape=[dim],
                default_initializer=paddle.nn.initializer.Constant(
                    layer_scale_init_value
                ),
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = math.floor(sequence_length**0.5)
        residual = x
        x = x.reshape([batch_size, input_dim, input_dim, hidden_size])
        x = x.transpose([0, 3, 1, 2])
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])
        x = self.norm(x, time)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape([batch_size, sequence_length, hidden_size])
        return residual + self.drop_path(x)


class ResNetBlock(paddle.nn.Layer):
    def __init__(self, config, dim):
        super().__init__()
        kernel_size = 3
        pad = (kernel_size - 1) // 2
        self.conv1 = paddle.nn.Conv2D(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.conv2 = paddle.nn.Conv2D(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = paddle.nn.BatchNorm2D(dim)
        self.bn2 = paddle.nn.BatchNorm2D(dim)

    def forward(self, x, time):
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = math.floor(sequence_length**0.5)
        residual = x
        x = x.reshape([batch_size, input_dim, input_dim, hidden_size])
        x = x.transpose([0, 3, 1, 2])
        x = self.conv1(x)
        x = self.bn1(x)
        x = paddle.nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape([batch_size, sequence_length, hidden_size])
        return x + residual


class ScOTPatchEmbeddings(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.projection = paddle.nn.Conv2D(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = paddle.compat.nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = paddle.compat.nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[paddle.Tensor]):
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose([0, 2, 1])
        return embeddings, output_dimensions


class ScOTEmbeddings(paddle.nn.Layer):
    def __init__(self, config, use_mask_token=False):
        super().__init__()
        self.patch_embeddings = ScOTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = (
            self.create_parameter(
                shape=[1, 1, config.embed_dim],
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if use_mask_token
            else None
        )
        self.position_embeddings = (
            self.create_parameter(
                shape=[1, num_patches, config.embed_dim],
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            if config.use_absolute_embeddings
            else None
        )
        layer_norm = ConditionalLayerNorm if config.use_conditioning else LayerNorm
        self.norm = layer_norm(config.embed_dim)
        self.dropout = paddle.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values, bool_masked_pos=None, time=None):
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings, time)
        batch_size, seq_len, _ = embeddings.shape
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand([batch_size, seq_len, embeddings.shape[-1]])
            mask = bool_masked_pos.unsqueeze(-1).astype(mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, output_dimensions


class ScOTLayer(paddle.nn.Layer):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path=0.0,
        shift_size=0,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.set_shift_and_window_size(input_resolution)
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=(
                pretrained_window_size
                if isinstance(pretrained_window_size, collections.abc.Iterable)
                else (pretrained_window_size, pretrained_window_size)
            ),
        )
        layer_norm = ConditionalLayerNorm if config.use_conditioning else LayerNorm
        self.layernorm_before = layer_norm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.layernorm_after = layer_norm(dim, eps=config.layer_norm_eps)
        self.attn_mask_cache = {}
        self.pad_cache = {}

    def set_shift_and_window_size(self, input_resolution):
        target_window_size = (
            self.window_size
            if isinstance(self.window_size, collections.abc.Iterable)
            else (self.window_size, self.window_size)
        )
        target_shift_size = (
            self.shift_size
            if isinstance(self.shift_size, collections.abc.Iterable)
            else (self.shift_size, self.shift_size)
        )
        window_dim = (
            int(input_resolution[0].item())
            if paddle.is_tensor(input_resolution[0])
            else input_resolution[0]
        )
        self.window_size = window_dim if window_dim <= target_window_size[0] else target_window_size[0]
        self.shift_size = (
            0
            if input_resolution
            <= (
                self.window_size
                if isinstance(self.window_size, collections.abc.Iterable)
                else (self.window_size, self.window_size)
            )
            else target_shift_size[0]
        )

    def get_attn_mask(self, height, width, dtype):
        cache_key = (height, width, self.shift_size, self.window_size, str(dtype))
        if cache_key in self.attn_mask_cache:
            return self.attn_mask_cache[cache_key]
        if self.shift_size > 0:
            img_mask = paddle.zeros([1, height, width, 1], dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape([-1, self.window_size * self.window_size])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = paddle.where(
                attn_mask != 0,
                paddle.full(attn_mask.shape, -100.0, dtype=attn_mask.dtype),
                paddle.zeros(attn_mask.shape, dtype=attn_mask.dtype),
            )
        else:
            attn_mask = None
        self.attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        cache_key = (height, width, self.window_size)
        if cache_key in self.pad_cache:
            pad_values = self.pad_cache[cache_key]
            if pad_values[3] > 0 or pad_values[5] > 0:
                hidden_states = paddle.compat.nn.functional.pad(hidden_states, pad_values)
            return hidden_states, pad_values
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        self.pad_cache[cache_key] = pad_values
        if pad_right > 0 or pad_bottom > 0:
            hidden_states = paddle.compat.nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_dimensions: Tuple[int, int],
        time: paddle.Tensor,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, ...]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states
        hidden_states = hidden_states.reshape([batch_size, height, width, channels])
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        if self.shift_size > 0:
            shifted_hidden_states = paddle.roll(
                hidden_states, shifts=(-self.shift_size, -self.shift_size), axis=[1, 2]
            )
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.reshape(
            [-1, self.window_size * self.window_size, channels]
        )
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]
        attention_windows = attention_output.reshape(
            [-1, self.window_size, self.window_size, channels]
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )
        if self.shift_size > 0:
            attention_windows = paddle.roll(
                shifted_windows, shifts=(self.shift_size, self.shift_size), axis=[1, 2]
            )
        else:
            attention_windows = shifted_windows
        if pad_values[3] > 0 or pad_values[5] > 0:
            attention_windows = attention_windows[:, :height, :width, :]
        attention_windows = attention_windows.reshape([batch_size, height * width, channels])
        hidden_states = shortcut + self.drop_path(self.layernorm_before(attention_windows, time))
        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(self.layernorm_after(layer_output, time))
        return (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)


class ScOTPatchRecovery(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_out_channels, hidden_size = config.num_out_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_out_channels = num_out_channels
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.projection = paddle.nn.Conv2DTranspose(
            in_channels=hidden_size,
            out_channels=num_out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mixup = paddle.nn.Conv2D(
            num_out_channels,
            num_out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias_attr=False,
        )

    def maybe_crop(self, pixel_values, height, width):
        if pixel_values.shape[2] > height:
            pixel_values = pixel_values[:, :, :height, :]
        if pixel_values.shape[3] > width:
            pixel_values = pixel_values[:, :, :, :width]
        return pixel_values

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose([0, 2, 1])
        hidden_states = hidden_states.reshape(
            [hidden_states.shape[0], hidden_states.shape[1], *self.grid_size]
        )
        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.image_size[0], self.image_size[1])
        return self.mixup(output)


class ScOTPatchMerging(paddle.nn.Layer):
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer=LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = paddle.nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = paddle.compat.nn.functional.pad(input_feature, pad_values)
        return input_feature

    def forward(self, input_feature, input_dimensions, time):
        height, width = input_dimensions
        batch_size, _, num_channels = input_feature.shape
        input_feature = input_feature.reshape([batch_size, height, width, num_channels])
        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = paddle.concat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], axis=-1
        )
        input_feature = input_feature.reshape([batch_size, -1, 4 * num_channels])
        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature, time)
        return input_feature


class ScOTPatchUnmerging(paddle.nn.Layer):
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer=LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.upsample = paddle.nn.Linear(dim, 2 * dim, bias_attr=False)
        self.mixup = paddle.nn.Linear(dim // 2, dim // 2, bias_attr=False)
        self.norm = norm_layer(dim // 2)

    def maybe_crop(self, input_feature, height, width):
        height_in, width_in = input_feature.shape[1], input_feature.shape[2]
        if height_in > height:
            input_feature = input_feature[:, :height, :, :]
        if width_in > width:
            input_feature = input_feature[:, :, :width, :]
        return input_feature

    def forward(self, input_feature, output_dimensions, time):
        output_height, output_width = output_dimensions
        batch_size, seq_len, hidden_size = input_feature.shape
        input_height = input_width = math.floor(seq_len**0.5)
        input_feature = self.upsample(input_feature)
        input_feature = input_feature.reshape(
            [batch_size, input_height, input_width, 2, 2, hidden_size // 2]
        )
        input_feature = input_feature.transpose([0, 1, 3, 2, 4, 5])
        input_feature = input_feature.reshape(
            [batch_size, 2 * input_height, 2 * input_width, hidden_size // 2]
        )
        input_feature = self.maybe_crop(input_feature, output_height, output_width)
        input_feature = input_feature.reshape([batch_size, -1, hidden_size // 2])
        input_feature = self.norm(input_feature, time)
        return self.mixup(input_feature)


class ScOTEncodeStage(paddle.nn.Layer):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        downsample,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        window_size = (
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = paddle.nn.LayerList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[i],
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            layer_norm = ConditionalLayerNorm if config.use_conditioning else LayerNorm
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=layer_norm)
        else:
            self.downsample = None

    def forward(
        self,
        hidden_states,
        input_dimensions,
        time,
        head_mask=None,
        output_attentions=False,
        always_partition=False,
    ):
        height, width = input_dimensions
        inputs = hidden_states
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
            )
            hidden_states = layer_outputs[0]
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling + inputs, input_dimensions, time
            )
        else:
            output_dimensions = (height, width, height, width)
        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTDecodeStage(paddle.nn.Layer):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        upsample,
        upsampled_size,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        window_size = (
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = paddle.nn.LayerList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[depth - 1 - i],
                    pretrained_window_size=pretrained_window_size,
                )
                for i in reversed(range(depth))
            ]
        )
        if upsample is not None:
            layer_norm = ConditionalLayerNorm if config.use_conditioning else LayerNorm
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=layer_norm)
            self.upsampled_size = upsampled_size
        else:
            self.upsample = None

    def forward(
        self,
        hidden_states,
        input_dimensions,
        time,
        head_mask=None,
        output_attentions=False,
        always_partition=False,
    ):
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
            )
            hidden_states = layer_outputs[0]
        hidden_states_before_upsampling = hidden_states
        if self.upsample is not None:
            height_upsampled, width_upsampled = self.upsampled_size
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.upsample(
                hidden_states_before_upsampling,
                (height_upsampled, width_upsampled),
                time,
            )
        else:
            output_dimensions = (height, width, height, width)
        stage_outputs = (
            hidden_states,
            hidden_states_before_upsampling,
            output_dimensions,
        )
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTEncoder(paddle.nn.Layer):
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = paddle.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths)
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[: drop_rates_encode_decode.shape[0] // 2]
        ]
        self.layers = paddle.nn.LayerList(
            [
                ScOTEncodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])
                    ],
                    downsample=ScOTPatchMerging if (i_layer < self.num_layers - 1) else None,
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in range(self.num_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        input_dimensions,
        time,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_hidden_states_before_downsampling=False,
        always_partition=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.reshape(
                [batch_size, *input_dimensions, hidden_size]
            ).transpose([0, 3, 1, 2])
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
            )
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.reshape(
                    [batch_size, output_dimensions[0], output_dimensions[1], hidden_size]
                ).transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.reshape(
                    [batch_size, *input_dimensions, hidden_size]
                ).transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[3:]
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )
        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOTDecoder(paddle.nn.Layer):
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = paddle.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths)
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[drop_rates_encode_decode.shape[0] // 2 :]
        ]
        self.layers = paddle.nn.LayerList(
            [
                ScOTDecodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[i_layer + 1 :]) : sum(config.depths[i_layer:])
                    ],
                    upsample=ScOTPatchUnmerging if i_layer > 0 else None,
                    upsampled_size=(
                        grid_size[0] // (2 ** (i_layer - 1)),
                        grid_size[1] // (2 ** (i_layer - 1)),
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in reversed(range(self.num_layers))
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        input_dimensions,
        skip_states,
        time,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_hidden_states_before_upsampling=False,
        always_partition=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.reshape(
                [batch_size, *input_dimensions, hidden_size]
            ).transpose([0, 3, 1, 2])
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if i != 0 and skip_states[len(skip_states) - i] is not None:
                hidden_states = hidden_states + skip_states[len(skip_states) - i]
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
            )
            hidden_states = layer_outputs[0]
            hidden_states_before_upsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            if output_hidden_states and output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states_before_upsampling.shape
                reshaped_hidden_state = hidden_states_before_upsampling.reshape(
                    [batch_size, output_dimensions[0], output_dimensions[1], hidden_size]
                ).transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states_before_upsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.reshape(
                    [batch_size, *input_dimensions, hidden_size]
                ).transpose([0, 3, 1, 2])
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[3:]
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )
        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOT(paddle.nn.Layer):
    def __init__(self, config, use_mask_token=False):
        super().__init__()
        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers_encoder - 1))
        self.embeddings = ScOTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ScOTEncoder(config, self.embeddings.patch_grid)
        self.decoder = ScOTDecoder(config, self.embeddings.patch_grid)
        self.patch_recovery = ScOTPatchRecovery(config)
        if config.residual_model == "convnext":
            res_model = ConvNeXtBlock
        elif config.residual_model == "resnet":
            res_model = ResNetBlock
        else:
            raise ValueError("residual_model must be 'convnext' or 'resnet'")
        self.residual_blocks = paddle.nn.LayerList(
            [
                (
                    paddle.nn.LayerList(
                        [res_model(config, config.embed_dim * 2**i) for _ in range(depth)]
                    )
                    if depth > 0
                    else paddle.nn.LayerList([paddle.nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )
        self.post_init()

    @property
    def dtype(self):
        parameters = list(self.parameters())
        return parameters[0].dtype if parameters else paddle.float32

    def _init_weights(self, module):
        if isinstance(module, (paddle.nn.Linear, paddle.nn.Conv2D, paddle.nn.Conv2DTranspose)):
            weight = getattr(module, "weight", None)
            if weight is not None:
                initializer = paddle.nn.initializer.Normal(
                    mean=0.0, std=self.config.initializer_range
                )
                initializer(weight)
            bias = getattr(module, "bias", None)
            if bias is not None:
                paddle.nn.initializer.Constant(0.0)(bias)
        elif isinstance(module, paddle.nn.LayerNorm):
            if getattr(module, "bias", None) is not None:
                paddle.nn.initializer.Constant(0.0)(module.bias)
            if getattr(module, "weight", None) is not None:
                paddle.nn.initializer.Constant(1.0)(module.weight)

    def post_init(self):
        for layer in self.sublayers(include_self=True):
            self._init_weights(layer)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(
                [num_hidden_layers, -1, -1, -1, -1]
            )
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {head_mask.ndim}"
        return head_mask.astype(self.dtype)

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config=None,
        ignore_mismatched_sizes=False,
    ):
        if config is None:
            config = ScOTConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        state_dict_path = os.path.join(
            pretrained_model_name_or_path, "model_state.pdparams"
        )
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"Missing Paddle weights: {state_dict_path}")
        loaded_state = paddle.load(state_dict_path)
        current_state = model.state_dict()
        filtered_state = {}
        unexpected_keys = []
        mismatched_keys = []
        for key, value in loaded_state.items():
            if key not in current_state:
                unexpected_keys.append(key)
                continue
            if list(value.shape) != list(current_state[key].shape):
                mismatched_keys.append(key)
                continue
            filtered_state[key] = value
        missing_keys = [key for key in current_state.keys() if key not in filtered_state]
        if (unexpected_keys or mismatched_keys or missing_keys) and not ignore_mismatched_sizes:
            raise ValueError(
                "Checkpoint keys do not match model. "
                f"Missing: {missing_keys[:10]}, unexpected: {unexpected_keys[:10]}, mismatched: {mismatched_keys[:10]}"
            )
        current_state.update(filtered_state)
        model.set_state_dict(current_state)
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        paddle.save(
            self.state_dict(), os.path.join(save_directory, "model_state.pdparams")
        )

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].attention.prune_heads(heads)
        for layer, heads in reversed(list(heads_to_prune.items())):
            self.decoder.layers[layer].attention.prune_heads(heads)

    def _downsample(self, image, target_size):
        image_size = image.shape[-2]
        freqs = paddle.fft.fftfreq(n=image_size, d=1 / image_size)
        sel = paddle.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = paddle.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        return paddle.fft.ifft2(image_hat, norm="forward").real()

    def _upsample(self, image, target_size):
        image_size = image.shape[-2]
        image_hat = paddle.fft.fft2(image, norm="forward")
        image_hat = paddle.fft.fftshift(image_hat)
        pad_size = (target_size - image_size) // 2
        real = paddle.compat.nn.functional.pad(
            image_hat.real(), (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        imag = paddle.compat.nn.functional.pad(
            image_hat.imag(), (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        image_hat = paddle.fft.ifftshift(paddle.complex(real, imag))
        return paddle.fft.ifft2(image_hat, norm="forward").real()

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        time: Optional[paddle.Tensor] = None,
        bool_masked_pos: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        pixel_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ScOTOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if pixel_values is None:
            raise ValueError("pixel_values cannot be None")

        head_mask = self.get_head_mask(
            head_mask, self.num_layers_encoder + self.num_layers_decoder
        )
        if isinstance(head_mask, list):
            head_mask_encoder = head_mask[: self.num_layers_encoder]
            head_mask_decoder = head_mask[self.num_layers_encoder :]
        else:
            head_mask_encoder, head_mask_decoder = head_mask.split(
                [self.num_layers_encoder, self.num_layers_decoder]
            )

        image_size = pixel_values.shape[2]
        if image_size != self.config.image_size:
            if image_size < self.config.image_size:
                pixel_values = self._upsample(pixel_values, self.config.image_size)
            else:
                pixel_values = self._downsample(pixel_values, self.config.image_size)

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, time=time
        )
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            time,
            head_mask=head_mask_encoder,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )
        if return_dict:
            skip_states = list(encoder_outputs.hidden_states[1:])
        else:
            skip_states = list(encoder_outputs[1][1:])

        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]:
                skip_states[i] = (
                    block(skip_states[i])
                    if isinstance(block, paddle.nn.Identity)
                    else block(skip_states[i], time)
                )

        input_dim = math.floor(skip_states[-1].shape[1] ** 0.5)
        decoder_output = self.decoder(
            skip_states[-1],
            (input_dim, input_dim),
            time=time,
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = (
            decoder_output.last_hidden_state if return_dict else decoder_output[0]
        )
        prediction = self.patch_recovery(sequence_output)
        if self.config.learn_residual:
            if self.config.num_channels > self.config.num_out_channels:
                pixel_values = pixel_values[:, 0 : self.config.num_out_channels]
            prediction = prediction + pixel_values
        if image_size != self.config.image_size:
            if image_size > self.config.image_size:
                prediction = self._upsample(prediction, image_size)
            else:
                prediction = self._downsample(prediction, image_size)
        if pixel_mask is not None and labels is not None:
            prediction = paddle.where(pixel_mask, labels.astype(prediction.dtype), prediction)

        loss = None
        if labels is not None:
            if self.config.p == 1:
                loss_fn = paddle.nn.functional.l1_loss
            elif self.config.p == 2:
                loss_fn = paddle.nn.functional.mse_loss
            else:
                raise ValueError("p must be 1 or 2")
            if self.config.channel_slice_list_normalized_loss is not None:
                loss = paddle.mean(
                    paddle.stack(
                        [
                            loss_fn(
                                prediction[
                                    :,
                                    self.config.channel_slice_list_normalized_loss[i] : self.config.channel_slice_list_normalized_loss[i + 1],
                                ],
                                labels[
                                    :,
                                    self.config.channel_slice_list_normalized_loss[i] : self.config.channel_slice_list_normalized_loss[i + 1],
                                ],
                            )
                            / (
                                loss_fn(
                                    labels[
                                        :,
                                        self.config.channel_slice_list_normalized_loss[i] : self.config.channel_slice_list_normalized_loss[i + 1],
                                    ],
                                    paddle.zeros_like(
                                        labels[
                                            :,
                                            self.config.channel_slice_list_normalized_loss[i] : self.config.channel_slice_list_normalized_loss[i + 1],
                                        ]
                                    ),
                                )
                                + 1e-10
                            )
                            for i in range(len(self.config.channel_slice_list_normalized_loss) - 1)
                        ]
                    )
                )
            else:
                loss = loss_fn(prediction, labels)

        if not return_dict:
            output = (prediction,) + decoder_output[1:] + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ScOTOutput(
            loss=loss,
            output=prediction,
            hidden_states=(
                decoder_output.hidden_states + encoder_outputs.hidden_states
                if output_hidden_states
                else None
            ),
            attentions=(
                decoder_output.attentions + encoder_outputs.attentions
                if output_attentions
                else None
            ),
            reshaped_hidden_states=(
                decoder_output.reshaped_hidden_states
                + encoder_outputs.reshaped_hidden_states
                if output_hidden_states
                else None
            ),
        )
