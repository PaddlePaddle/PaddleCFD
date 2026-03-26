from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.kd_random_horizontal_flip import KDRandomHorizontalFlip
from kappadata.transforms.kd_random_resized_crop import KDRandomResizedCrop
from .norm.kd_image_net_norm import KDImageNetNorm


class ImagenetMinaugTransform(KDComposeTransform):
    def __init__(self, size=224, interpolation="bicubic", min_scale=0.08):
        super().__init__(transforms=[
            KDRandomResizedCrop(size=size, interpolation=interpolation, scale=(min_scale, 1.)),
            KDRandomHorizontalFlip(),
            KDImageNetNorm(),
        ])
