import paddle


class LearnedBatchNorm(paddle.nn.Layer):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.dim = dim
        self.affine = affine
        self.mean = paddle.nn.Parameter(paddle.zeros(*self._shape()))
        self.logvar = paddle.nn.Parameter(paddle.zeros(*self._shape()))
        if affine:
            self.weight = paddle.nn.Parameter(paddle.ones(*self._shape()))
            self.bias = paddle.nn.Parameter(paddle.zeros(*self._shape()))
        else:
            self.weight = None
            self.bias = None

    def _shape(self):
        return 1, self.dim

    def forward(self, x):
        return (x - self.mean) / self.logvar.exp() * self.weight + self.bias


class LearnedBatchNorm1d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1


class LearnedBatchNorm2d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1, 1


class LearnedBatchNorm3d(LearnedBatchNorm):
    def _shape(self):
        return 1, self.dim, 1, 1, 1
