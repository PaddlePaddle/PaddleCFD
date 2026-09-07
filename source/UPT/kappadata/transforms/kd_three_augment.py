from .base.kd_stochastic_transform import KDStochasticTransform
from .kd_gaussian_blur_pil import KDGaussianBlurPIL
from .kd_grayscale import KDGrayscale
from .kd_solarize import KDSolarize


class KDThreeAugment(KDStochasticTransform):
    def __init__(
            self,
            threshold,
            sigma,
            kernel_size=None,
            blur_kind="pil",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.grayscale = KDGrayscale()
        self.solarize = KDSolarize(threshold=threshold)
        if blur_kind in ["pil", "pillow"]:
            self.gaussian_blur = KDGaussianBlurPIL(sigma=sigma)
        elif blur_kind in ["tv", "paddlevision"]:
            assert kernel_size is not None
            self.gaussian_blur = KDGaussianBlurPIL(sigma=sigma, kernel_size=kernel_size)
        else:
            raise NotImplementedError

    def set_rng(self, rng):
        self.gaussian_blur.set_rng(rng)
        return super().set_rng(rng)

    def __call__(self, x, ctx=None):
        choice = self.rng.integers(3)
        if choice == 0:
            return self.grayscale(x)
        if choice == 1:
            return self.solarize(x)
        if choice == 2:
            return self.gaussian_blur(x)
        raise NotImplementedError
