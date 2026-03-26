from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDMagnitudeJitter(KDStochasticTransform):
    def __init__(self, alpha, inplace=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.inplace = inplace

    def __call__(self, x, ctx=None):
        # jitter with random magnitude (np.random.beta(10, 10) has mean=0.5 -> magnitude is in [0.5, 1.5])
        # beta distribution is not guaranteed to have mean=0.5 but range should nevertheless be in [0.5, 1.5]
        magnitude = self.rng.beta(self.alpha, self.alpha) + 0.5
        return x.mul_(magnitude) if self.inplace else x * magnitude
