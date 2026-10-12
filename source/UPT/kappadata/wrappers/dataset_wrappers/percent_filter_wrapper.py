import numpy as np

from kappadata.datasets.kd_subset import KDSubset


class PercentFilterWrapper(KDSubset):
    def __init__(self, dataset, from_percent=None, to_percent=None, ceil_from_index=False, ceil_to_index=False):
        self.from_percent = from_percent or 0.
        self.to_percent = to_percent or 1.
        assert self.from_percent is None or 0. <= self.from_percent <= 1.
        assert self.to_percent is None or 0. <= self.to_percent <= 1.
        self.ceil_from_index = ceil_from_index
        self.ceil_to_index = ceil_to_index

        self.from_index = self.from_percent * len(dataset)
        self.from_index = np.ceil(self.from_index) if self.ceil_from_index else int(self.from_index)

        self.to_index = self.to_percent * len(dataset)
        self.to_index = np.ceil(self.to_index) if self.ceil_to_index else int(self.to_index)

        indices = np.arange(self.from_index, self.to_index, dtype=np.int64)
        super().__init__(dataset=dataset, indices=indices)
