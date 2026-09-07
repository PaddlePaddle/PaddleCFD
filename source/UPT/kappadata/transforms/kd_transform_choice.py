from .kd_stochastic_transform import KDStochasticTransform


class KDTransformChoice(KDStochasticTransform):
    def __init__(self, transforms, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, x, ctx=None):
        # select which transform to apply
        idx = int(self.rng.random() * len(self.transforms))
        # apply transform
        x = self.transforms[idx](x, ctx=ctx)

        # update context
        if ctx is not None:
            ctx["transform_choice"] = idx

        return x