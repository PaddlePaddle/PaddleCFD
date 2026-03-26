from kappadata.compat.image import InterpolationMode, resolve_interpolation
from paddle.vision.transforms import Resize

from .base.kd_stochastic_transform import KDStochasticTransform
from .kd_random_crop import KDRandomCrop


class KDSimpleRandomCrop(KDStochasticTransform):
    def __init__(
            self,
            size,
            padding=4,
            interpolation="bicubic",
            padding_mode="reflect",
            fill=0,
            pad_if_needed=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # Resize doesn't support interpolation argument as string
        self.resize = Resize(size=size, interpolation=resolve_interpolation(interpolation), **kwargs)
        self.random_crop = KDRandomCrop(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
        )

    def set_rng(self, rng):
        self.random_crop.set_rng(rng)
        return super().set_rng(rng)

    def __call__(self, x, ctx=None):
        x = self.resize(x)
        return self.random_crop(x, ctx=ctx)
