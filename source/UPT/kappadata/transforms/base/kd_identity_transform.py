from .kd_transform import KDTransform


class KDIdentityTransform(KDTransform):
    def __call__(self, x, ctx=None):
        return x
