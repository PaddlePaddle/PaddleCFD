from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDSidNorm(KDImageNorm):
    def __init__(self, **kwargs):
        # values from AudioMAE Table 3 https://arxiv.org/abs/2207.06405
        super().__init__(mean=(-6.370,), std=(3.074,), **kwargs)
