from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_additive_gaussian_noise import KDAdditiveGaussianNoise


class KDRandomAdditiveGaussianNoise(KDRandomApplyBase):
    def __init__(
            self,
            std: float,
            magnitude: float = 1.,
            magnitude_std: float = float("inf"),
            magnitude_min: float = 0.,
            magnitude_max: float = 1.,
            clip_min: float = None,
            clip_max: float = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise = KDAdditiveGaussianNoise(
            std=std,
            magnitude=magnitude,
            magnitude_std=magnitude_std,
            magnitude_min=magnitude_min,
            magnitude_max=magnitude_max,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    def _scale_strength(self, factor):
        self.noise.scale_strength(factor)

    def forward(self, x, ctx):
        return self.noise(x, ctx=ctx)
