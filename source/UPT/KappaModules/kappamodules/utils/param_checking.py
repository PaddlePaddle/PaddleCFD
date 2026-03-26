import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(repeat(x, n))

    return parse


def _is_ntuple(n):
    def check(x):
        return isinstance(x, tuple) and len(param) == n

    return check


def to_ntuple(x, n):
    return _ntuple(n=n)(x)


def is_ntuple(x, n):
    return _is_ntuple(n=n)(x)


to_2tuple = _ntuple(2)
is_2tuple = _is_ntuple(2)
