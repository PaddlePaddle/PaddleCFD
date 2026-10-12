from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDCifar10Norm(KDImageNorm):
    def __init__(self, **kwargs):
        super().__init__(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), **kwargs)
