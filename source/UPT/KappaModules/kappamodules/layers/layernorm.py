import einops
import paddle


class LayerNorm1d(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = paddle.nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "batch_size dim height -> batch_size height dim")
        x = self.layer(x)
        x = einops.rearrange(x, "batch_size height dim -> batch_size dim height")
        return x


class LayerNorm2d(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = paddle.nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(
            x, "batch_size dim height width -> batch_size height width dim"
        )
        x = self.layer(x)
        x = einops.rearrange(
            x, "batch_size height width dim -> batch_size dim height width"
        )
        return x


class LayerNorm3d(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'eps' in kwargs:
            kwargs['epsilon'] = kwargs.pop('eps')
        self.layer = paddle.nn.LayerNorm(*args, **kwargs)

    def forward(self, x):
        x = einops.rearrange(
            x, "batch_size dim height width depth -> batch_size height width depth dim"
        )
        x = self.layer(x)
        x = einops.rearrange(
            x, "batch_size height width depth dim -> batch_size dim height width depth"
        )
        return x
