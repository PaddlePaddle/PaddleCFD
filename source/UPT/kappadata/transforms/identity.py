from .base.kd_transform import KDTransform


class Identity(KDTransform):
    def __call__(self, x, ctx=None):
        return x
