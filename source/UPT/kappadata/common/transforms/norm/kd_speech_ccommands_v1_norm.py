from kappadata.transforms.norm.kd_image_norm import KDImageNorm


class KDSpeechCommandsV1Norm(KDImageNorm):
    def __init__(self, **kwargs):
        # values from AudioMAE Table 3 https://arxiv.org/abs/2207.06405
        super().__init__(mean=(-6.702,), std=(5.448,), **kwargs)
