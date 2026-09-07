import paddle


class Identity(paddle.nn.Identity):
    """
    nn.Identity but ignores all passed arguments
    useful for replacing normalization layers with identity without worrying about the passed arguments
    Examples:
        >>> if norm_mode == "batchnorm":
        >>>   norm_ctor = nn.BatchNorm1d
        >>> elif norm_mode == "none":
        >>>   norm_ctor = Identity
        >>> else:
        >>>   raise NotImplemented
        >>> ...
        >>> self.norm = norm_ctor(dim)
    """

    def __init__(self, *_, **__):
        super().__init__()

    def forward(self, x, *_, **__):
        return x
