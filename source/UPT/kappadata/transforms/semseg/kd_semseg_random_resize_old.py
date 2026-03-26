import paddle
from kappadata.compat.image import InterpolationMode, resolve_interpolation
from paddle.vision.transforms.functional import resize

from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform
from kappadata.utils.param_checking import to_2tuple


class KDSemsegRandomResizeOld(KDStochasticTransform):
    """ resize image and mask to base_size * ratio where ratio is randomly sampled """

    def __init__(self, base_size, ratio, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.base_size = to_2tuple(base_size)
        self.ratio = to_2tuple(ratio)
        self.interpolation = resolve_interpolation(interpolation)

    def __call__(self, xsemseg, ctx=None):
        new_size = self.get_params()
        x, semseg = xsemseg
        x = resize(x, new_size, self.interpolation)
        squeeze_semseg = False
        if paddle.is_tensor(semseg):
            semseg = semseg.unsqueeze(0)
            squeeze_semseg = True
        semseg = resize(semseg, new_size, InterpolationMode.NEAREST)
        if squeeze_semseg:
            semseg = semseg.squeeze(0)
        return x, semseg

    def get_params(self):
        ratio = self.rng.uniform(self.ratio[0], self.ratio[1])
        height, width = self.base_size
        new_height = round(height * ratio)
        new_width = round(width * ratio)
        return new_height, new_width
