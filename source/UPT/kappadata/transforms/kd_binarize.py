import paddle
from paddle.vision.transforms.functional import to_tensor

from .base import KDTransform


class KDBinarize(KDTransform):
    def __call__(self, x, ctx=None):
        if not paddle.is_tensor(x):
            x = to_tensor(x)
        return (x > 0.5).float()
