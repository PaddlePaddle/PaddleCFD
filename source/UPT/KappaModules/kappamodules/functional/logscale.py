import paddle


def to_logscale(x):
    return paddle.sign(x) * paddle.log1p(x=x.abs())


def from_logscale(x):
    return paddle.sign(x) * (x.abs().exp() - 1)
