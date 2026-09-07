import paddle


class DropPath(paddle.nn.Sequential):
    """
    Efficiently drop paths (Stochastic Depth) per sample such that dropped samples are not processed.
    This is a subclass of nn.Sequential and can be used either as standalone Module or like nn.Sequential.
    Examples::
        >>> # use as nn.Sequential module
        >>> sequential_droppath = DropPath(nn.Linear(4, 4), drop_prob=0.2)
        >>> y = sequential_droppath(paddle.randn(10, 4))

        >>> # use as standalone module
        >>> standalone_layer = nn.Linear(4, 4)
        >>> standalone_droppath = DropPath(drop_prob=0.2)
        >>> y = standalone_droppath(paddle.randn(10, 4), standalone_layer)
    """

    def __init__(
        self,
        *args,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
        stochastic_drop_prob: bool = False,
        drop_prob_tolerance: float = 0.01,
    ):
        super().__init__(*args)
        assert 0.0 <= drop_prob < 1.0
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob
        self.drop_prob_tolerance = drop_prob_tolerance

    @property
    def drop_prob(self):
        return self._drop_prob

    @drop_prob.setter
    def drop_prob(self, value):
        assert 0.0 <= value < 1.0
        self._drop_prob = value

    @property
    def keep_prob(self):
        return 1.0 - self.drop_prob

    def forward(
        self,
        x,
        residual_path=None,
        residual_path_kwargs=None,
        residual=None,
    ):
        if residual is not None:
            assert len(self) == 0 and residual_path is None
        else:
            assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0.0 or not self.training:
            if residual is not None:
                return x + residual
            elif residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        keep_count = max(int(bs * self.keep_prob), 1)
        actual_keep_prob = keep_count / bs
        drop_path_delta = self.keep_prob - actual_keep_prob
        if self.stochastic_drop_prob or drop_path_delta > self.drop_prob_tolerance:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            # random_tensor = x.new_empty(shape).bernoulli_(p=self.keep_prob)
            random_tensor = (paddle.rand(shape, dtype="float32") < self.keep_prob).cast(x.dtype)
            if self.scale_by_keep:
                # random_tensor.div_(self.keep_prob)
                random_tensor = random_tensor / self.keep_prob
            if residual is not None:
                return x + residual * random_tensor
            elif residual_path is None:
                return x + super().forward(x, **residual_path_kwargs) * random_tensor
            else:
                return x + residual_path(x, **residual_path_kwargs) * random_tensor
        scale = bs / keep_count
        perm = paddle.randperm(bs)[:keep_count]
        if self.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.0
        residual_path_kwargs = {
            key: (value[perm] if paddle.is_tensor(value) else value)
            for key, value in residual_path_kwargs.items()
        }
        if residual is not None:
            residual = residual[perm]
        elif residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        x_flat = paddle.flatten(x, start_axis=1)
        residual_flat = paddle.flatten(residual.cast(x.dtype), start_axis=1)
        x_flat = paddle.index_add(
            x=x_flat,
            index=perm,
            axis=0,
            value=residual_flat * alpha,
        )
        return paddle.reshape(x_flat, shape=x.shape)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
