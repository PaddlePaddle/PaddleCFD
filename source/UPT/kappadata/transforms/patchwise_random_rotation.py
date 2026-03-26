from paddle.vision.transforms.functional import rotate

from .base.kd_stochastic_transform import KDStochasticTransform


class PatchwiseRandomRotation(KDStochasticTransform):
    def __call__(self, x, ctx=None):
        c, l, patch_h, patch_w = x.shape
        # sample rotations
        rotations = self.rng.integers(0, 4, size=l) * 90
        if ctx is not None:
            ctx["patchwise_random_rotation"] = rotations
        # rotate patches
        for i in range(l):
            x[:, i] = rotate(x[:, i], angle=float(rotations[i]))
        return x
