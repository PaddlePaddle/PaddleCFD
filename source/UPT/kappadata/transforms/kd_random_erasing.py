import math

import numpy as np
import paddle

from .base.kd_random_apply_base import KDRandomApplyBase


class KDRandomErasing(KDRandomApplyBase):
    """
    adaption of timm.data.random_erasing.RandomErasing
    additional features:
    - deterministic behavior by setting seed
    removed features:
    - support for batchwise apply (intended to be used during dataloading, not during model forward pass)
    """

    def __init__(
            self,
            min_area=0.02,
            max_area=1 / 3,
            min_aspect=0.3,
            max_aspect=None,
            mode="zeros",
            min_count=1,
            max_count=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.mode = mode.lower()
        if self.mode == "zeros":
            self._get_replacement = self._get_replacement_zeros
        elif self.mode == "channelwise":
            self._get_replacement = self._get_replacement_channelwise
        elif self.mode == "pixelwise":
            self._get_replacement = self._get_replacement_pixelwise
        else:
            raise NotImplementedError(f"mode '{self.mode}' not supported (use zeros, channelwise or pixelwise)")

    @staticmethod
    def _get_replacement_zeros(c, h, w):
        return paddle.zeros(c, h, w)

    # noinspection PyUnusedLocal
    def _get_replacement_channelwise(self, c, h, w):
        return paddle.to_tensor(self.rng.standard_normal(size=(c, 1, 1), dtype=np.float32))

    def _get_replacement_pixelwise(self, c, h, w):
        return paddle.to_tensor(self.rng.standard_normal(size=(c, h, w), dtype=np.float32))

    def forward(self, x, ctx):
        # sample how many rectangles to erase
        if self.min_count == self.max_count:
            n_rects = self.min_count
        else:
            n_rects = int(self.rng.integers(self.min_count, self.max_count))

        c, img_h, img_w = x.shape
        area_per_rect = img_h * img_w / n_rects
        for _ in range(n_rects):
            # try 10 times -> skip if not successful
            for _ in range(10):
                target_area = self.rng.uniform(self.min_area, self.max_area) * area_per_rect
                aspect_ratio = math.exp(self.rng.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = int(self.rng.integers(0, img_h - h + 1))
                    left = int(self.rng.integers(0, img_w - w + 1))
                    x[:, top:top + h, left:left + w] = self._get_replacement(c=c, h=h, w=w)
                    break
        return x
