import paddle


class LayerScale(paddle.nn.Layer):
    def __init__(self, dim: int, init_scale: float = 1e-05):
        super().__init__()
        if init_scale is None:
            self.gamma = None
        else:
            self.gamma = paddle.nn.Parameter(
                paddle.full(size=(dim,), fill_value=init_scale)
            )

    def forward(self, x):
        if self.gamma is None:
            return x
        return x * self.gamma
