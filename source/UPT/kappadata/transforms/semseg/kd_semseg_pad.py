from kappadata.compat.image import pad, get_image_size

from kappadata.transforms.base.kd_transform import KDTransform
from kappadata.utils.param_checking import to_2tuple


class KDSemsegPad(KDTransform):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)

    def __call__(self, xsemseg, ctx=None):
        x, semseg = xsemseg
        width, height = get_image_size(x)
        pad_h = max(0, self.size[0] - height)
        pad_w = max(0, self.size[1] - width)
        pad_top = pad_bot = pad_h // 2
        if pad_h % 2 == 1:
            pad_bot += 1
        pad_left = pad_right = pad_w // 2
        if pad_w % 2 == 1:
            pad_right += 1

        padding = (pad_left, pad_top, pad_right, pad_bot)
        x = pad(x, padding=padding, fill=0)
        semseg = pad(semseg, padding=padding, fill=-1)
        return x, semseg
