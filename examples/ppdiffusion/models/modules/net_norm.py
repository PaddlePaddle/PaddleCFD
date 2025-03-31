import paddle


class LayerNorm(paddle.nn.Layer):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=[1, dim, 1, 1]))
        self.b = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=[1, dim, 1, 1]))

    def forward(self, x):
        var = paddle.var(x=x, axis=1, unbiased=False, keepdim=True)
        mean = paddle.mean(x=x, axis=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(paddle.nn.Layer):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class PostNorm(paddle.nn.Layer):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x):
        x = self.fn(x)
        return self.norm(x)
