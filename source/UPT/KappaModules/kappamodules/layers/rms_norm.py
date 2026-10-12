import paddle

from kappamodules.utils.shapes import to_ndim


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.g = paddle.nn.Parameter(paddle.ones(dim))

    def forward(self, x):
        return (
            paddle.nn.functional.normalize(x, dim=1)
            * to_ndim(self.g.view(1, -1), ndim=x.ndim)
            * x.shape[1] ** 0.5
        )
