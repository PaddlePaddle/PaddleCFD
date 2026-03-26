import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.global_rng import GlobalRng
from kappadata.utils.getall_as_tensor import getall_as_numpy


class SwapLabelWrapper(KDWrapper):
    def __init__(self, dataset, p, seed=0, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        assert 0. <= p <= 1.
        rng = GlobalRng() if seed is None else np.random.default_rng(seed=seed)
        num_classes = self.dataset.getdim_class()
        self.apply = rng.random(size=(len(dataset),)) < p
        og_classes = getall_as_numpy(dataset, item="class")
        new_classes = rng.integers(low=0, high=num_classes, size=(len(dataset),))
        self.classes = np.where(self.apply, new_classes, og_classes)

    @property
    def _global_rng(self):
        return GlobalRng()

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return int(self.classes[idx])

    def getall_class(self):
        return self.classes.tolist()

    # noinspection PyUnusedLocal
    def getitem_apply(self, idx, ctx=None):
        return bool(self.apply[idx])

    def getall_apply(self):
        return self.apply[idx].tolist()