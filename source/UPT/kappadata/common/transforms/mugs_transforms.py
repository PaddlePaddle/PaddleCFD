import kappadata.transforms as kdt
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from .norm.kd_image_net_norm import KDImageNetNorm


class MUGSStrongTransform(KDComposeTransform):
    def __init__(self, size, min_scale=0.08, max_scale=1.0):
        super().__init__(transforms=[
            kdt.KDRandAugment(
                num_ops=2,
                magnitude=9,
                magnitude_std=0.5,
                interpolation="random",
                fill_color=(124, 116, 104),
            ),
            kdt.KDRandomResizedCrop(size=size, scale=(min_scale, max_scale), interpolation="bicubic"),
            kdt.KDRandomHorizontalFlip(),
            KDImageNetNorm(),
            kdt.KDRandomErasing(p=0.25, mode="pixelwise", max_count=1),
        ])


class MUGSStrongGlobalTransform(MUGSStrongTransform):
    def __init__(self, size=224, min_scale=0.25):
        super().__init__(size=size, min_scale=min_scale)


class MUGSStrongLocalTransform(MUGSStrongTransform):
    def __init__(self, size=96, min_scale=0.05, max_scale=0.25):
        super().__init__(size=size, min_scale=min_scale, max_scale=max_scale)
