from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDRoll(KDStochasticTransform):
    def __call__(self, x, ctx=None):
        # used for random left/right stero channel permutation and random start time
        # can be used for arbitrary shapes
        shifts = [self.rng.integers(dim) if dim > 1 else 0 for dim in x.shape]
        return x.roll(shifts=shifts, dims=list(range(x.ndim)))
