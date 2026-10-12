from paddle.vision.transforms import Resize
from kappadata.compat.image import InterpolationMode, resolve_interpolation

from kappadata.utils.image_utils import get_dimensions
from .base import KDTransform


class KDMinsize(KDTransform):
    """
    if image is smaller than minimal size:
        resize to minimal size
    else:
        pass
    """

    def __init__(self, ctx_prefix=None, interpolation="bilinear", **kwargs):
        super().__init__(ctx_prefix=ctx_prefix)
        self.resize = Resize(interpolation=resolve_interpolation(interpolation), **kwargs)

    def __call__(self, x, ctx=None):
        _, h, w = get_dimensions(x)
        if isinstance(self.resize.size, int):
            if h >= self.resize.size and w >= self.resize.size:
                return x
        elif len(self.resize.size) == 2:
            if h >= self.resize.size[0] and w >= self.resize.size[1]:
                return x
        return self.resize(x)
