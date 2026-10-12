import paddle


class Normalize(paddle.nn.Layer):
    def __init__(self, dim=1, p=2.0):
        super().__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return paddle.nn.functional.normalize(x, dim=self.dim, p=self.p)
