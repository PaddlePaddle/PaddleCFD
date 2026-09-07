from kappadata.compat.data import get_worker_info

from kappadata.utils.random import get_rng_from_global


class KDTransform:
    def __init__(self, ctx_prefix: str = None):
        self.ctx_prefix = ctx_prefix or type(self).__name__
        # sanity check to avoid accidentally overwriting base method
        assert type(self).scale_strength == KDTransform.scale_strength
        assert type(self).worker_init_fn == KDTransform.worker_init_fn

    @property
    def is_deterministic(self):
        return True

    def set_rng(self, rng):
        pass

    @property
    def is_kd_transform(self):
        return True

    def __call__(self, x, ctx=None):
        raise NotImplementedError

    def worker_init_fn(self, rank, **kwargs):
        # if num_workers == 0 -> get_worker_info() is None
        # if num_workers the worker_init_fn can be called manually although this is not recommended
        # e.g. the counter of a scheduled transform will be a global object and therefore no longer be correct
        info = get_worker_info()
        if info is None:
            num_workers = 1
        else:
            num_workers = info.num_workers
        # problem: since rngs are initialized in the __init__ methods they are copied when workers are spawned
        # solution: overwrite the rng when workers are spawned
        self.set_rng(get_rng_from_global())
        self._worker_init_fn(rank, num_workers, **kwargs)

    def _worker_init_fn(self, rank, num_workers, **kwargs):
        pass

    def scale_strength(self, factor):
        assert 0. <= factor <= 1.
        self._scale_strength(factor)

    def _scale_strength(self, factor):
        pass

    @classmethod
    def supports_scale_strength(cls):
        return cls._scale_strength != KDTransform._scale_strength
