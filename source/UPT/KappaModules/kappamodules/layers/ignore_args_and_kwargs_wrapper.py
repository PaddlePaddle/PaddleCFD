import paddle


class IgnoreArgsAndKwargsWrapper(paddle.nn.Layer):
    """
    wrapper for any nn.Module that ignores everything besides the first arg passed to forward
    use-case: pytorch_geometrics InstanceNorm
    self.message = km.Sequential(
      km.IgnoreArgsAndKwargsWrapper(nn.Linear(10, 10)),
      pyg.InstanceNorm(10),
      ...
    )
    y = self.message(x, batch=batch)
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *_, **__):
        return self.module(x)
