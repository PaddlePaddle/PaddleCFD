from kappadata.factory import object_to_transform
from .kd_transform import KDTransform


class KDComposeTransform(KDTransform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = [object_to_transform(transform) for transform in transforms]

    def _worker_init_fn(self, rank, num_workers, **kwargs):
        for t in self.transforms:
            if isinstance(t, KDTransform):
                t._worker_init_fn(rank, num_workers, **kwargs)

    @property
    def is_deterministic(self):
        return all(t.is_deterministic for t in self.transforms)

    @property
    def is_kd_transform(self):
        return all(isinstance(t, KDTransform))

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            if isinstance(x, (list, tuple)):
                # apply for each sample
                x = [self._apply(t, xx, ctx) for xx in x]
                # flatten outputs in case they are a list (i.e. avoid list of lists)
                flat = []
                for xx in x:
                    if isinstance(xx, (list, tuple)):
                        flat += xx
                    else:
                        flat.append(xx)
                x = flat
            else:
                # apply to one sample
                x = self._apply(t, x, ctx)
        return x

    @staticmethod
    def _apply(t, x, ctx):
        if isinstance(t, KDTransform):
            return t(x, ctx)
        return t(x)

    def set_rng(self, rng):
        for t in self.transforms:
            if isinstance(t, KDTransform):
                t.set_rng(rng)
        return self

    def _scale_strength(self, factor):
        for t in self.transforms:
            if isinstance(t, KDTransform):
                t.scale_strength(factor)
