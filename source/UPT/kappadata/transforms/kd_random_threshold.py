from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_threshold import KDThreshold


class KDRandomThreshold(KDRandomApplyBase):
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
        self.threshold = KDThreshold(
            threshold=threshold,
            threshold_std=threshold_std,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            mode=mode,
        )

    def _scale_strength(self, factor):
        self.threshold.scale_strength(factor)

    def forward(self, x, ctx):
        return self.threshold(x, ctx=ctx)
