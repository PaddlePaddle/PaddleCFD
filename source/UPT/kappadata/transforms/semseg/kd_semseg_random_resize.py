import paddle
from kappadata.compat.image import InterpolationMode, resolve_interpolation
from paddle.vision.transforms.functional import resize

from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform
from kappadata.utils.param_checking import to_2tuple


class KDSemsegRandomResize(KDStochasticTransform):
    """ resize image and mask to base_size * ratio where ratio is randomly sampled """

    def __init__(self, base_size, ratio, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.ratio_resolution = to_2tuple(base_size)
        self.ratio_range = to_2tuple(ratio)
        self.interpolation = resolve_interpolation(interpolation)

    def __call__(self, xsemseg, ctx=None):
        x, seg = xsemseg

        # get params
        if paddle.is_tensor(x):
            _, h, w = x.shape
        else:
            h = x.height
            w = x.width
        suggested_height, suggested_width = self.get_params()
        # scale by smallest scaleing while keeping aspect ratio
        max_long_edge = max(suggested_height, suggested_width)
        max_short_edge = min(suggested_height, suggested_width)
        smallest_scale = min(max_long_edge / max(h, w), max_short_edge / min(h, w))

        new_height = round(h * smallest_scale)
        new_width = round(w * smallest_scale)
        new_size = new_height, new_width

        x = resize(x, size=new_size, interpolation=self.interpolation)
        squeeze = False
        if paddle.is_tensor(seg):
            seg = seg.unsqueeze(0)
            squeeze = True
        seg = resize(seg, size=new_size, interpolation=InterpolationMode.NEAREST)
        if squeeze:
            seg = seg.squeeze(0)
        return x, seg

    def get_params(self):
        ratio_min, ratio_max = self.ratio_range
        ratio = self.rng.uniform(ratio_min, ratio_max)
        height, width = self.ratio_resolution
        suggested_height = height * ratio
        suggested_width = width * ratio
        return suggested_height, suggested_width
