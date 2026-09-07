from kappadata.utils.magnitude_sampler import MagnitudeSampler
from .base.kd_stochastic_transform import KDStochasticTransform


class KDThreshold(KDStochasticTransform):
    def __init__(
            self,
            threshold: float,
            threshold_std: float = 0.,
            threshold_min: float = 0.,
            threshold_max: float = 1.,
            mode="zeros",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.magnitude_sampler = MagnitudeSampler(
            magnitude=threshold,
            magnitude_std=threshold_std,
            magnitude_min=threshold_min,
            magnitude_max=threshold_max,
        )
        self.mode = mode
        self.ctx_key = f"{self.ctx_prefix}.threshold"

    def _scale_strength(self, factor):
        self.magnitude_sampler.scale_strength(factor)

    def __call__(self, x, ctx=None):
        magnitude = self.magnitude_sampler.sample(self.rng)
        if self.mode == "zeros":
            x[x < magnitude] = 0
        else:
            raise NotImplementedError
        return x
