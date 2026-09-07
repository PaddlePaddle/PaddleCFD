from kappadata.transforms.norm.kd_image_norm import KDImageNorm

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class KDImageNetNorm(KDImageNorm):
    def __init__(self, **kwargs):
        super().__init__(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, **kwargs)
