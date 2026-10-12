import paddle

from kappadata.utils.magnitude_sampler import MagnitudeSampler
from .base.kd_stochastic_transform import KDStochasticTransform


class KDAdditiveUniformNoise(KDStochasticTransform):
    def __init__(
            self,
            magnitude: float = 1.,
            magnitude_std: float = float("inf"),
            magnitude_min: float = 0.,
            magnitude_max: float = 1.,
            clip_min: float = None,
            clip_max: float = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.magnitude_sampler = MagnitudeSampler(
            magnitude=magnitude,
            magnitude_std=magnitude_std,
            magnitude_min=magnitude_min,
            magnitude_max=magnitude_max,
        )
        self.ctx_key = f"{self.ctx_prefix}.magnitude"
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _scale_strength(self, factor):
        self.magnitude_sampler.scale_strength(factor)

    def __call__(self, x, ctx=None):
        magnitude = self.magnitude_sampler.sample(self.rng)
        noise = paddle.to_tensor(self.rng.random(size=x.shape)).float() * magnitude
        if ctx is not None:
            ctx[self.ctx_key] = magnitude
        x = x + noise
        if self.clip_min is not None or self.clip_max is not None:
            x = paddle.clip(x, self.clip_min, self.clip_max)
        return x
