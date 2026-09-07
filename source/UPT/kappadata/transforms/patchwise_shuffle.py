from .base.kd_stochastic_transform import KDStochasticTransform


class PatchwiseShuffle(KDStochasticTransform):
    def __call__(self, x, ctx=None):
        c, l, patch_h, patch_w = x.shape
        # sample rotations
        permutation = self.rng.permutation(l)
        if ctx is not None:
            ctx["permutation"] = permutation
        x = x[:, permutation]
        return x
