import paddle.vision.transforms.functional as F

from .base.kd_transform import KDTransform


class KDSolarize(KDTransform):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = self.og_threshold = threshold
        self.ctx_key = f"{self.ctx_prefix}.threshold"

    def _scale_strength(self, factor):
        # PIL -> threshold >= 256 -> no augmentation
        # tensor -> threshold >= 1. -> no augmentation
        if isinstance(self.og_threshold, int):
            # PIL
            self.threshold = int(256 - (256 - self.og_threshold) * factor)
        elif isinstance(self.og_threshold, float):
            # tensor
            self.threshold = 1. - (1. - self.og_threshold) * factor
        else:
            raise NotImplementedError

    def __call__(self, x, ctx=None):
        if ctx is not None:
            ctx[self.ctx_key] = self.threshold
        return F.solarize(x, self.threshold)
