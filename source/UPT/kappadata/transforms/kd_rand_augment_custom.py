from .kd_rand_augment import KDRandAugment


class KDRandAugmentCustom(KDRandAugment):
    """
    KDRandAugment but sample without replacement and make posterize not produce fully black images
    """

    def _sample_transforms(self):
        return self.rng.choice(self.ops, size=self.num_ops, replace=False)

    def posterize(self, x, magnitude):
        # if magnitude >= 1.0 --> black image
        magnitude = min(0.99, magnitude)
        return super().posterize(x, magnitude)
