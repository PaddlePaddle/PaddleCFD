import collections.abc
from itertools import repeat


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(repeat(x, n))

    return parse


def to_ntuple(x, n):
    return _ntuple(n=n)(x)


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)
