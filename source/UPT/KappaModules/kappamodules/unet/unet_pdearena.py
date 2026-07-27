import paddle


class ResidualBlock(paddle.nn.Layer):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.activation = paddle.nn.GELU()
        self.conv1 = paddle.nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv2 = paddle.nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        if in_channels != out_channels:
            self.shortcut = paddle.nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1)
            )
        else:
            self.shortcut = paddle.nn.Identity()
        if norm:
            self.norm1 = paddle.nn.GroupNorm(n_groups, in_channels)
            self.norm2 = paddle.nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = paddle.nn.Identity()
            self.norm2 = paddle.nn.Identity()
        self.cond_emb = paddle.compat.nn.Linear(
            cond_channels, 2 * out_channels if use_scale_shift_norm else out_channels
        )
        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.init.zeros_(self.conv2.weight)
        paddle.nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(self.activation(self.norm1(x)))
        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = paddle.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
            h = self.conv2(self.activation(h))
        else:
            h = h + emb_out
            h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(paddle.nn.Layer):
    """Attention block This is similar to [transformer multi-head
    attention](https://arxiv.org/abs/1706.03762).

    Args:
        n_channels: the number of channels in the input
        n_heads:  the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups: the number of groups for [group normalization][paddle.nn.GroupNorm]

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k=None, n_groups: int = 1):
        """ """
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = paddle.nn.GroupNorm(n_groups, n_channels)
        self.projection = paddle.compat.nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = paddle.compat.nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = paddle.chunk(qkv, 3, dim=-1)
        attn = paddle.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = paddle.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(paddle.nn.Layer):
    """Down block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool): Whether to use attention block
        norm (bool): Whether to use normalization
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            cond_channels,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = paddle.nn.Identity()

    def forward(self, x, emb):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class UpBlock(paddle.nn.Layer):
    """Up block This combines `ResidualBlock` and `AttentionBlock`.

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool): Whether to use attention block
        norm (bool): Whether to use normalization
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            cond_channels,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = paddle.nn.Identity()

    def forward(self, x, emb) -> paddle.Tensor:
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class MiddleBlock(paddle.nn.Layer):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        cond_channels (int): Number of channels in the conditioning vector.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        norm (bool, optional): Whether to use normalization. Defaults to False.
        use_scale_shift_norm (bool, optional): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
    """

    def __init__(
        self,
        n_channels: int,
        cond_channels: int,
        has_attn: bool = False,
        norm: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.attn = AttentionBlock(n_channels) if has_attn else paddle.nn.Identity()
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            cond_channels,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )

    def forward(self, x, emb) -> paddle.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


class Upsample(paddle.nn.Layer):
    """Scale up the feature map by $2 \\times$"""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = paddle.nn.Conv2DTranspose(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(paddle.nn.Layer):
    """Scale down the feature map by $\\frac{1}{2} \\times$"""

    def __init__(self, n_channels):
        super().__init__()
        self.conv = paddle.nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x):
        return self.conv(x)


class UnetPdearena(paddle.nn.Layer):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        input_dim (int): Number input dimensions
        output_dim (int): Number of output dimensions
        hidden_channels (int): Number of channels in the hidden layers
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        param_conditioning (Optional[str]): Type of conditioning to use. Defaults to None.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers

    Note:
        Currently, only `scalar` parameter conditioning is supported.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels,
        cond_dim: int = None,
        norm: bool = False,
        ch_mults=(1, 2, 2, 4),
        is_attn=(False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use_scale_shift_norm: bool = False,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.activation = paddle.nn.GELU()
        n_resolutions = len(ch_mults)
        n_channels = hidden_channels
        if use1x1:
            self.image_proj = paddle.nn.Conv2d(input_dim, n_channels, kernel_size=1)
        else:
            self.image_proj = paddle.nn.Conv2d(
                input_dim, n_channels, kernel_size=(3, 3), padding=(1, 1)
            )
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        cond_dim,
                        has_attn=is_attn[i],
                        norm=norm,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = paddle.nn.ModuleList(down)
        self.middle = MiddleBlock(
            out_channels,
            cond_dim,
            has_attn=mid_attn,
            norm=norm,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        cond_dim,
                        has_attn=is_attn[i],
                        norm=norm,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    cond_dim,
                    has_attn=is_attn[i],
                    norm=norm,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = paddle.nn.ModuleList(up)
        if norm:
            self.norm = paddle.nn.GroupNorm(8, n_channels)
        else:
            self.norm = paddle.nn.Identity()
        if use1x1:
            self.final = paddle.nn.Conv2d(in_channels, output_dim, kernel_size=1)
        else:
            self.final = paddle.nn.Conv2d(
                in_channels, output_dim, kernel_size=(3, 3), padding=(1, 1)
            )
        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.init.zeros_(self.final.weight)
        paddle.nn.init.zeros_(self.final.bias)

    def forward(self, x, emb):
        assert x.ndim == 4
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            else:
                x = m(x, emb)
            h.append(x)
        x = self.middle(x, emb)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                s = h.pop()
                x = paddle.cat((x, s), dim=1)
                x = m(x, emb)
        x = self.final(self.activation(self.norm(x)))
        return x
