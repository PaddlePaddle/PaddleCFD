from .base.kd_random_apply_base import KDRandomApplyBase
from .kd_color_jitter import KDColorJitter


class KDRandomColorJitter(KDRandomApplyBase):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        super().__init__(**kwargs)
        self.color_jitter = KDColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            ctx_prefix=self.ctx_prefix,
        )

    def set_rng(self, rng):
        self.color_jitter.set_rng(rng)
        return super().set_rng(rng)

    def _populate_ctx_on_skip(self, ctx):
        ctx[self.color_jitter.ctx_key_fn_idx] = [-1, -1, -1, -1]
        ctx[self.color_jitter.ctx_key_brightness] = -1.
        ctx[self.color_jitter.ctx_key_contrast] = -1.
        ctx[self.color_jitter.ctx_key_saturation] = -1.
        ctx[self.color_jitter.ctx_key_hue] = -1.

    def _scale_strength(self, factor):
        self.color_jitter.scale_strength(factor)

    def forward(self, x, ctx):
        return self.color_jitter(x, ctx)
