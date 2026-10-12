from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDAudioSetNorm(KDImageNorm):
    def __init__(self, **kwargs):
        # values from AudioMAE Table 3 https://arxiv.org/abs/2207.06405
        super().__init__(mean=(-4.268,), std=(4.569,), **kwargs)
