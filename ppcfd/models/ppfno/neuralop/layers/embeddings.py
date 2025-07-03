import paddle


class PositionalEmbedding(paddle.nn.Layer):

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = paddle.arange(start=0, end=self.num_channels // 2, dtype=
            'float32')
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(y=freqs.to(x.dtype))
        x = paddle.concat(x=[x.cos(), x.sin()], axis=1)
        return x
