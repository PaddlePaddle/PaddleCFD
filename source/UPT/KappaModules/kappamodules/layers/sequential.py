import paddle


class Sequential(paddle.nn.Sequential):
    """
    paddle.nn.Sequential but one can pass arbitrary arguments
    net = nn.Sequential(...)
    net(a, b) -> this fails

    net = km.Sequential(...)
    net(a, b) -> passes a and b to all layers in km.Sequential
    """

    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x
