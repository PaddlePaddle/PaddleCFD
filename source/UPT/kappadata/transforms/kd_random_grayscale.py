import paddle.vision.transforms.functional as F

from .base.kd_random_apply_base import KDRandomApplyBase


class KDRandomGrayscale(KDRandomApplyBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.og_p = self.p

    def _scale_strength(self, factor):
        self.p = self.og_p * factor

    def forward(self, x, ctx):
        # if ctx is not None:
        #     ctx["random_grayscale"] = True
        return F.rgb_to_grayscale(x, num_output_channels=F.get_image_num_channels(x))
