import paddle


class MLP(paddle.nn.Layer):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
        n_layers=2, n_dim=2, non_linearity=paddle.nn.functional.gelu,
        dropout=0.0, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = (in_channels if out_channels is None else
            out_channels)
        self.hidden_channels = (in_channels if hidden_channels is None else
            hidden_channels)
        self.non_linearity = non_linearity
        self.dropout = paddle.nn.LayerList(sublayers=[paddle.nn.Dropout(p=
            dropout) for _ in range(n_layers)]) if dropout > 0.0 else None
        Conv = getattr(paddle.nn, f'Conv{n_dim}D')
        self.fcs = paddle.nn.LayerList()
        for i in range(n_layers):
            if i == 0 and i == n_layers - 1:
                self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1)
                    )
            elif i == n_layers - 1:
                self.fcs.append(Conv(self.hidden_channels, self.
                    out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.
                    hidden_channels, 1))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x


class MLPLinear(paddle.nn.Layer):

    def __init__(self, layers, non_linearity=paddle.nn.functional.gelu,
        dropout=0.0):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.fcs = paddle.nn.LayerList()
        self.non_linearity = non_linearity
        self.dropout = paddle.nn.LayerList(sublayers=[paddle.nn.Dropout(p=
            dropout) for _ in range(self.n_layers)]) if dropout > 0.0 else None
        for j in range(self.n_layers):
            self.fcs.append(paddle.nn.Linear(in_features=layers[j],
                out_features=layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x
