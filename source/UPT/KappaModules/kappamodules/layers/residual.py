import paddle


class Residual(paddle.nn.Sequential):
    """
    Wrapper for nn.Sequential for residual connections.
    Useful to include residual connections inside other sequentials
    Examples::
        >>> # manual residual connection
        >>> layer = nn.Linear(4, 4)
        >>> manual_y = x + layer(x)
        >>> # with Residual
        >>> layer = Residual(nn.Linear(4, 4))
        >>> residual_y = layer(x)

        >>> # block with 2 residual layers
        >>> layer0 = nn.Linear(4, 4)
        >>> layer1 = nn.Linear(4, 4)
        >>> manual_y = x + layer0(x)
        >>> manual_y = manual_y + layer1(manual_y)
        >>> # with Residual
        >>> block = nn.Sequential(Residual(nn.Linear(4, 4)), Residual(4, 4))
        >>> y = block(x)
    """

    def forward(self, x):
        return x + super().forward(x)
