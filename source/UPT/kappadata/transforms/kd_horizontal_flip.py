from paddle.vision.transforms.functional import hflip

from .base.kd_transform import KDTransform


class KDHorizontalFlip(KDTransform):
    def __call__(self, x, ctx=None):
        return hflip(x)
