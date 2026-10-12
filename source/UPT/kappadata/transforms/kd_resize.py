from paddle.vision.transforms import Resize
from kappadata.compat.image import InterpolationMode, resolve_interpolation

from .base import KDTransform


class KDResize(KDTransform):
    """ wrapper for paddle.vision.transforms.Resize as it doesn't support passing a string as interpolation """

    def __init__(self, *args, ctx_prefix=None, interpolation="bilinear", **kwargs):
        super().__init__(ctx_prefix=ctx_prefix)
        self.resize = Resize(*args, interpolation=resolve_interpolation(interpolation), antialias=True, **kwargs)

    def __call__(self, x, ctx=None):
        return self.resize(x)
