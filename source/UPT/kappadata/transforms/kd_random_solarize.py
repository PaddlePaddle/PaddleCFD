from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_solarize import KDSolarize


class KDRandomSolarize(KDRandomApplyBase):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.solarize = KDSolarize(threshold=threshold, ctx_prefix=self.ctx_prefix)

    def _scale_strength(self, factor):
        self.solarize.scale_strength(factor)

    def _populate_ctx_on_skip(self, ctx):
        ctx[self.solarize.ctx_key] = -1

    def forward(self, x, ctx):
        return self.solarize(x, ctx=ctx)
