import logging

from paddle.io import Dataset


class CachedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.logger = logging.getLogger(type(self).__name__)
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self._cached_getitem(idx)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def _cached_getitem(self, index):
        raise NotImplementedError

    def dispose(self):
        raise NotImplementedError
