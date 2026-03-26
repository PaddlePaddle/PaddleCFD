import numpy as np
from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.one_hot import to_one_hot_vector


class SemiWrapper(KDWrapper):
    def __init__(self, semi_percent, seed=0, **kwargs):
        super().__init__(**kwargs)
        assert 0. <= semi_percent <= 1.
        rng = np.random.default_rng(seed=seed)
        self.semi_idxs = set(rng.permutation(len(self))[:int(len(self) * semi_percent)])

    def getitem_class(self, idx, ctx=None):
        if idx in self.semi_idxs:
            return -1
        return self.dataset.getitem_class(idx, ctx=ctx)

    def getall_class(self):
        cls = self.dataset.getall_class()
        for idx in self.semi_idxs:
            cls[idx] = -1
        return cls