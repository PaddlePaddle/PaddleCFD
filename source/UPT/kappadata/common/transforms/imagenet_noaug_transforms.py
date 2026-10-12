from paddle.vision.transforms import CenterCrop

from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_resize import KDResize
from .norm.kd_image_net_norm import KDImageNetNorm


class ImagenetNoaugTransform(KDComposeTransform):
    def __init__(self, resize_size=256, center_crop_size=224, interpolation="bicubic"):
        assert resize_size >= center_crop_size
        super().__init__(transforms=[
            KDResize(size=resize_size, interpolation=interpolation),
            CenterCrop(size=center_crop_size),
            KDImageNetNorm(),
        ])
