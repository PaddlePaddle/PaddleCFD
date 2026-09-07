from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDCifar100Norm(KDImageNorm):
    def __init__(self, **kwargs):
        super().__init__(mean=(0.5071, 0.4867, 0.4408), std=(0.2673, 0.2564, 0.2762), **kwargs)
