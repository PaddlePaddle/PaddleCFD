from kappadata.compat.image import get_image_size, pad, crop

from kappadata.utils.param_checking import to_2tuple
from .base.kd_stochastic_transform import KDStochasticTransform


class KDRandomCrop(KDStochasticTransform):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, ctx=None):
        img = self._pad_image(img)
        i, j, h, w = self.get_params(img)
        if ctx is not None:
            ctx["random_crop"] = dict(i=i, j=j, h=h, w=w)
        return crop(img, i, j, h, w)

    def _pad_image(self, img):
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)

        width, height = get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = pad(img, padding, self.fill, self.padding_mode)

        return img

    def get_params(self, img):
        w, h = get_image_size(img)
        th, tw = self.size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = int(self.rng.integers(0, h - th + 1))
        j = int(self.rng.integers(0, w - tw + 1))
        return i, j, th, tw
