import paddle


class GlobalResponseNorm(paddle.nn.Layer):
    """adapted from timm.layers.grn.GlobalResponseNorm"""

    def __init__(self, dim, eps=1e-06, ndim=None):
        super().__init__()
        self.eps = eps
        if ndim is None:
            self.spatial_dim = 1, 2
            self.channel_dim = -1
            self.wb_shape = 1, 1, 1, -1
        elif ndim == 1:
            self.spatial_dim = (2,)
            self.channel_dim = 1
            self.wb_shape = 1, -1, 1
        elif ndim == 2:
            self.spatial_dim = 2, 3
            self.channel_dim = 1
            self.wb_shape = 1, -1, 1, 1
        elif ndim == 3:
            self.spatial_dim = 2, 3, 4
            self.channel_dim = 1
            self.wb_shape = 1, -1, 1, 1, 1
        else:
            raise NotImplementedError
        # self.weight = paddle.nn.Parameter(paddle.zeros(dim))
        # self.bias = paddle.nn.Parameter(paddle.zeros(dim))
        self.weight = self.create_parameter(
            shape=[dim],
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.bias = self.create_parameter(
            shape=[dim],
            default_initializer=paddle.nn.initializer.Constant(0.0),
            is_bias=True # 标记为 bias
        )

    def forward(self, x):
        # x_g = x.norm(p=2, axis=self.spatial_dim, keepdim=True)
        x_g = paddle.sqrt(paddle.sum(paddle.square(x), axis=self.spatial_dim, keepdim=True))
        x_n = x_g / (x_g.mean(axis=self.channel_dim, keepdim=True) + self.eps)
        return x + paddle.add(
            self.bias.view(self.wb_shape),
            1 * self.weight.view(self.wb_shape) * (x * x_n),
        )


class GlobalResponseNorm1d(GlobalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=1)


class GlobalResponseNorm2d(GlobalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=2)


class GlobalResponseNorm3d(GlobalResponseNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ndim=3)
