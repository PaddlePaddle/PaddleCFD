import paddle


class ParamlessBatchNorm1d(paddle.nn.Layer):
    """
    non-affine BatchNorm1d layer that doesn't need a dimension but also can't be used in eval mode
    this layer works with SyncBatchnorm
    """

    def __init__(self, num_features=None):
        super().__init__()
        self.norm = paddle.nn.BatchNorm1D(
            num_features=num_features, weight_attr=False, bias_attr=False
        )

    def forward(self, x):
        if not self.norm.track_running_stats:
            assert self.training
        return self.norm(x)
