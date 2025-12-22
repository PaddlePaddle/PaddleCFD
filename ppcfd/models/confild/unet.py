import math
from abc import abstractmethod

import numpy as np
import paddle

from .diffusion import split


###################### UNET Model #######################
def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return paddle.nn.Conv1D(*args, **kwargs)
    elif dims == 2:
        return paddle.nn.Conv2D(*args, **kwargs)
    elif dims == 3:
        return paddle.nn.Conv3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return paddle.nn.Linear(*args, **kwargs)


class TimestepBlock(paddle.nn.Layer):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        pass


class ResBlock(TimestepBlock):
    """
    Residual block with timestep embedding for diffusion models.

    Implements a residual connection with two convolutional layers, timestep conditioning,
    and optional up/downsampling. Supports FiLM-style adaptive normalization.

    Args:
        channels (int): Number of input channels.
        emb_channels (int): Number of timestep embedding channels.
        dropout (float): Dropout probability.
        out_channels (int, optional): Number of output channels. Defaults to channels.
        use_conv (bool, optional): Use conv for skip connection if channels differ. Defaults to False.
        use_scale_shift_norm (bool, optional): Use FiLM-style conditioning. Defaults to False.
        dims (int, optional): Spatial dimensions (1D/2D/3D). Defaults to 2.
        use_checkpoint (bool, optional): Use gradient checkpointing. Defaults to False.
        up (bool, optional): Apply upsampling. Defaults to False.
        down (bool, optional): Apply downsampling. Defaults to False.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = paddle.nn.Sequential(
            normalization(channels),
            paddle.nn.Silu(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = paddle.nn.Identity()
        self.emb_layers = paddle.nn.Sequential(
            paddle.nn.Silu(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = paddle.nn.Sequential(
            normalization(self.out_channels),
            paddle.nn.Silu(),
            paddle.nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = paddle.nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(tuple(emb_out.shape)) < len(tuple(h.shape)):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            (scale, shift) = paddle.chunk(emb_out, chunks=2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class TimestepEmbedSequential(paddle.nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


NUM_CLASSES = 1000


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return paddle.nn.AvgPool1D(*args, **kwargs, exclusive=False)
    elif dims == 2:
        return paddle.nn.AvgPool2D(*args, **kwargs, exclusive=False)
    elif dims == 3:
        return paddle.nn.AvgPool3D(*args, **kwargs, exclusive=False)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(paddle.nn.Layer):
    """
    Spatial downsampling layer (2x reduction).

    Can use either strided convolution or average pooling for downsampling.

    Args:
        channels (int): Number of input channels.
        use_conv (bool): Use strided conv (True) or avg pooling (False).
        dims (int, optional): Spatial dimensions. Defaults to 2.
        out_channels (int, optional): Number of output channels. Defaults to channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """Apply downsampling."""
        assert tuple(x.shape)[1] == self.channels
        return self.op(x)


class Upsample(paddle.nn.Layer):
    """
    Spatial upsampling layer (2x expansion).

    Uses nearest-neighbor interpolation followed by optional convolution.

    Args:
        channels (int): Number of input channels.
        use_conv (bool): Apply convolution after upsampling.
        dims (int, optional): Spatial dimensions. Defaults to 2.
        out_channels (int, optional): Number of output channels. Defaults to channels.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        """Apply upsampling."""
        assert tuple(x.shape)[1] == self.channels
        if self.dims == 3:
            x = paddle.nn.functional.interpolate(
                x,
                size=(tuple(x.shape)[2], tuple(x.shape)[3] * 2, tuple(x.shape)[4] * 2),
                mode="nearest",
            )
        else:
            x = paddle.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


def count_flops_attn(model, _x, y):
    b, c, *spatial = tuple(y[0].shape)
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += paddle.to_tensor(data=[matmul_ops], dtype="float64")


class QKVAttentionLegacy(paddle.nn.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = tuple(qkv.shape)
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        # split_size: When int, torch represents block size, paddle represents number of blocks
        (q, k, v) = split(qkv.reshape((bs * self.n_heads, ch * 3, length)), ch, 1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = paddle.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = paddle.nn.functional.softmax(weight.astype(dtype="float32"), axis=-1).astype(weight.dtype)
        a = paddle.einsum("bts,bcs->bct", weight, v)
        return a.reshape((bs, -1, length))

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(paddle.nn.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = tuple(qkv.shape)
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        (q, k, v) = qkv.chunk(chunks=3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = paddle.einsum(  # Non-complex
            "bct,bcs->bts",
            (q * scale).reshape([bs * self.n_heads, ch, length]),
            (k * scale).reshape([bs * self.n_heads, ch, length]),
        )
        weight = paddle.nn.functional.softmax(weight.astype(dtype="float32"), axis=-1).astype(weight.dtype)
        a = paddle.einsum("bts,bcs->bct", weight, v.reshape((bs * self.n_heads, ch, length)))
        return a.reshape((bs, -1, length))

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class GroupNorm32(paddle.nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.astype(dtype="float32")).astype(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with paddle.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        new_inputs = []
        for x in ctx.input_tensors:
            x.stop_gradient = False
            new_inputs.append(x)
        ctx.input_tensors = new_inputs
        with paddle.enable_grad():
            shallow_copies = [x.reshape(x.shape) for x in ctx.input_tensors]
            # print(shallow_copies)
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = paddle.grad(
            outputs=output_tensors,
            inputs=ctx.input_tensors + ctx.input_params,
            grad_outputs=output_grads,
            allow_unused=True,
            # retain_graph=True, create_graph=False
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        return tuple(input_grads)


class AttentionBlock(paddle.nn.Layer):
    """
    Self-attention block for spatial feature maps.

    Applies multi-head self-attention over spatial locations in feature maps,
    allowing the model to capture long-range dependencies.

    Args:
        channels (int): Number of input/output channels.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        num_head_channels (int, optional): Channels per head (overrides num_heads). Defaults to -1.
        use_checkpoint (bool, optional): Use gradient checkpointing. Defaults to False.
        use_new_attention_order (bool, optional): Use optimized attention implementation. Defaults to False.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = tuple(x.shape)
        x = x.reshape((b, c, -1))
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape((b, c, *spatial))


def convert_module_to_f16(module):
    if isinstance(module, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        module.weight.data = module.weight.data.astype(dtype="float16")
        if module.bias is not None:
            module.bias.data = module.bias.data.astype(dtype="float16")


def convert_module_to_f32(module):
    if isinstance(module, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        module.weight.data = module.weight.data.astype(dtype="float32")
        if module.bias is not None:
            module.bias.data = module.bias.data.astype(dtype="float32")


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings for diffusion models.

    Similar to positional encodings in transformers, but for continuous timesteps.
    Uses sinusoids of exponentially increasing frequencies.

    Args:
        timesteps (paddle.Tensor): Timestep values of shape (batch_size,).
        dim (int): Embedding dimension.
        max_period (int, optional): Maximum period for sinusoids. Defaults to 10000.

    Returns:
        paddle.Tensor: Timestep embeddings of shape (batch_size, dim).
    """
    half = dim // 2
    freqs = paddle.exp(-math.log(max_period) * paddle.arange(start=0, end=half, dtype="float32") / half)
    args = timesteps[:, None].astype(dtype="float32") * freqs[None]
    embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
    if dim % 2:
        embedding = paddle.concat([embedding, paddle.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class UNetModel(paddle.nn.Layer):
    """
    Full UNet model with attention and timestep embedding for diffusion models.

    Implements a U-Net architecture with residual blocks, self-attention at multiple resolutions,
    and timestep conditioning via adaptive normalization (FiLM). Designed for denoising diffusion
    probabilistic models (DDPM) and can be conditioned on class labels.

    Reference:
        Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
        Dhariwal & Nichol "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021)

    Args:
        image_size (int): Input image size (maintained for interface compatibility).
        in_channels (int): Number of channels in input tensor.
        model_channels (int): Base channel count for model (multiplied by channel_mult).
        out_channels (int): Number of channels in output tensor.
        num_res_blocks (int): Number of residual blocks per downsampling level.
        attention_resolutions (list/tuple): Downsample factors where to apply attention (e.g., [4, 8, 16]).
        dropout (float, optional): Dropout probability in residual blocks. Defaults to 0.0.
        channel_mult (tuple, optional): Channel multipliers per level (e.g., (1, 2, 4, 8)). Defaults to (1, 2, 4, 8).
        conv_resample (bool, optional): Use learned convolutional up/downsampling. Defaults to True.
        dims (int, optional): Data dimensionality (1=1D, 2=2D, 3=3D). Defaults to 2.
        num_classes (int, optional): Number of classes for class-conditional generation. Defaults to None.
        use_checkpoint (bool, optional): Enable gradient checkpointing to save memory. Defaults to False.
        use_fp16 (bool, optional): Use float16 precision for forward pass. Defaults to False.
        num_heads (int, optional): Number of attention heads in each attention block. Defaults to 1.
        num_head_channels (int, optional): Fixed channels per head (overrides num_heads if set). Defaults to -1.
        num_heads_upsample (int, optional): Attention heads for upsampling blocks. Defaults to -1 (use num_heads).
        use_scale_shift_norm (bool, optional): Use FiLM-style conditioning in ResBlocks. Defaults to False.
        resblock_updown (bool, optional): Use ResBlocks for up/downsampling instead of conv layers. Defaults to False.
        use_new_attention_order (bool, optional): Use optimized QKV attention implementation. Defaults to False.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> model = ppsci.arch.UNetModel(
        ...     image_size=64,
        ...     in_channels=3,
        ...     model_channels=128,
        ...     out_channels=3,
        ...     num_res_blocks=2,
        ...     attention_resolutions=[8, 16],
        ...     channel_mult=(1, 2, 4, 8),
        ...     num_heads=4,
        ... )
        >>> x = paddle.randn([4, 3, 64, 64])
        >>> t = paddle.randint(0, 1000, [4])
        >>> out = model(x, t)
        >>> print(out.shape)
        [4, 3, 64, 64]
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # Handle None channel_mult from config files
        if channel_mult is None:
            channel_mult = (1, 2, 4, 8)
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = paddle.nn.Sequential(
            linear(model_channels, time_embed_dim),
            paddle.nn.Silu(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = paddle.nn.Embedding(num_embeddings=self.num_classes, embedding_dim=time_embed_dim)
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = paddle.nn.LayerList(
            sublayers=[TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = []
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.output_blocks = paddle.nn.LayerList(sublayers=[])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = []
                layers.append(
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = paddle.nn.Sequential(
            normalization(ch),
            paddle.nn.Silu(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert tuple(y.shape) == (tuple(x.shape)[0],)
            emb = emb + self.label_emb(y)

        h = x.astype(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = paddle.concat([h, hs.pop()], axis=1)
            h = module(h, emb)

        h = h.astype(x.dtype)
        return self.out(h)
