from paddle.vision.transforms.functional import hflip

from kappadata.transforms.base.kd_random_apply_base import KDRandomApplyBase


class KDSemsegRandomHorizontalFlip(KDRandomApplyBase):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(p=p, **kwargs)

    def forward(self, xsemseg, ctx):
        # if ctx is not None:
        #     ctx["random_hflip"] = True
        x, semseg = xsemseg
        x = hflip(x)
        semseg = hflip(semseg)
        return x, semseg
