import paddle


class ContinuousSincosEmbed(paddle.nn.Layer):
    def __init__(self, dim, ndim, max_wavelength: int = 10000, dtype=paddle.float32):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        self.register_buffer(
            "omega",
            1.0
            / max_wavelength
            ** (
                paddle.arange(0, effective_dim_per_wave, 2, dtype=dtype)
                / effective_dim_per_wave
            ),
        )

    def forward(self, coords):
        out_dtype = coords.dtype
        ndim = coords.shape[-1]
        assert self.ndim == ndim
        out = coords.unsqueeze(-1).cast(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = paddle.concat([paddle.sin(out), paddle.cos(out)], axis=-1)
        if coords.ndim == 3:
            emb = paddle.flatten(emb, start_axis=2, stop_axis=3)
        elif coords.ndim == 2:
            emb = paddle.flatten(emb, start_axis=1, stop_axis=2)
        else:
            raise NotImplementedError
        emb = emb.cast(out_dtype)
        if self.padding > 0:
            padding = paddle.zeros([*emb.shape[:-1], self.padding], dtype=emb.dtype)
            emb = paddle.concat([emb, padding], axis=-1)
        return emb

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self).__name__}(dim={self.dim})"
