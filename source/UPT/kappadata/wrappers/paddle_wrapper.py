from functools import partial

from kappadata.datasets.kd_dataset import KDDataset
from .mode_wrapper import ModeWrapper


class PaddleWrapper(KDDataset):
    def __init__(self, dataset, mode):
        super().__init__()
        self.dataset = dataset
        self.mode = mode

    def __getattr__(self, item):
        if item.startswith('getitem_'):
            item = item[len('getitem_'):]
            assert ModeWrapper.has_item(mode=self.mode, item=item)
            item_idx = ModeWrapper.get_item_index(mode=self.mode, item=item)
            return partial(self._getitem, item_idx=item_idx)
        return getattr(self.dataset, item)

    def _getitem(self, idx, ctx=None, item_idx=None):
        batch = self.dataset[idx]
        return batch[item_idx]

    def __len__(self):
        return len(self.dataset)
