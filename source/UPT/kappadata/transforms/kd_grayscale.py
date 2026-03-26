import paddle.vision.transforms.functional as F

from .base import KDTransform


class KDGrayscale(KDTransform):
    def __call__(self, x, ctx=None):
        return F.rgb_to_grayscale(x, num_output_channels=F.get_image_num_channels(x))
