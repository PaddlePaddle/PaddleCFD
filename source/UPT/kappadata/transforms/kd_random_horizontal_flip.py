from paddle.vision.transforms.functional import hflip

from .base.kd_random_apply_base import KDRandomApplyBase


class KDRandomHorizontalFlip(KDRandomApplyBase):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(p=p, **kwargs)

    def forward(self, x, ctx):
        # if ctx is not None:
        #     ctx["random_hflip"] = True
        return hflip(x)
