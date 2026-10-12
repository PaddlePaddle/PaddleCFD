import kappadata.transforms as kdt


class MAEFinetuneTransform(kdt.KDComposeTransform):
    def __init__(self):
        super().__init__(transforms=[
            kdt.KDRandomResizedCrop(size=224, interpolation="bicubic"),
            kdt.KDRandomHorizontalFlip(),
            kdt.KDRandAugment(
                num_ops=2,
                magnitude=9,
                magnitude_std=0.5,
                interpolation="bicubic",
                fill_color=(124, 116, 104),
            ),
            kdt.KDImageNetNorm(),
            kdt.KDRandomErasing(p=0.25, mode="pixelwise", max_count=1),
        ])
