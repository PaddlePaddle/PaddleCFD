import paddle


class MLP(paddle.nn.Layer):

    def __init__(self, layers, nonlinearity, out_nonlinearity=None,
        normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = paddle.nn.LayerList()
        for j in range(self.n_layers):
            self.layers.append(paddle.nn.Linear(in_features=layers[j],
                                                out_features=layers[j + 1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(paddle.nn.BatchNorm1D(num_features=
                        layers[j + 1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class PositionalEmbedding(paddle.nn.Layer):

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = paddle.arange(start=0, end=self.num_channels // 2, dtype='float32')
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs

        freqs = paddle.cast(freqs, x.dtype)
        x = x.outer(y=freqs)
        x = paddle.concat(x=[x.cos(), x.sin()], axis=1)
        return x


class AdaIN(paddle.nn.Layer):

    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-05):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps
        if mlp is None:
            mlp = paddle.nn.Sequential(paddle.nn.Linear(in_features=
                embed_dim, out_features=512), paddle.nn.GELU(), paddle.nn.
                Linear(in_features=512, out_features=2 * in_channels))
        self.mlp = mlp
        self.embedding = None

    def update_embeddding(self, x):
        self.embedding = x.reshape((self.embed_dim,))

    def forward(self, x):
        assert self.embedding is not None, 'AdaIN: update embeddding before running forward'
        
        x_mlp = self.mlp(self.embedding)
        num_or_sections = x_mlp.shape[0] // self.in_channels
        weight, bias = paddle.split(
            x=x_mlp, num_or_sections=num_or_sections, axis=0
        )
        
        # weight, bias = paddle.split(x=self.mlp(self.embedding),
            # num_or_sections=self.in_channels, axis=0)
        return paddle.nn.functional.group_norm(x=x, num_groups=self.
            in_channels, weight=weight, bias=bias, epsilon=self.eps)


class Projection(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
        n_dim=2, non_linearity=paddle.nn.functional.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (in_channels if hidden_channels is None else
            hidden_channels)
        self.non_linearity = non_linearity
        Conv = getattr(paddle.nn, f'Conv{n_dim}D')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x
