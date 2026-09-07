from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDEsc50Norm(KDImageNorm):
    def __init__(self, **kwargs):
        # values from AudioMAE Table 3 https://arxiv.org/abs/2207.06405
        super().__init__(mean=(-6.627,), std=(5.359,), **kwargs)
