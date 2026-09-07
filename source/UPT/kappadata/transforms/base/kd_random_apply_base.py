from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform


class KDRandomApplyBase(KDStochasticTransform):
    def __init__(self, p, **kwargs):
        super().__init__(**kwargs)
        assert 0. <= p <= 1.
        self.p = p

    def __call__(self, x, ctx=None):
        apply = self.rng.random() < self.p
        if apply:
            return self.forward(x, ctx)
        if ctx is not None:
            self._populate_ctx_on_skip(ctx)
        return x

    def _populate_ctx_on_skip(self, ctx):
        # TODO this should be mandatory
        pass

    def forward(self, x, ctx):
        raise NotImplementedError
