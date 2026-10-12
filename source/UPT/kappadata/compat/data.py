import math
from collections.abc import Mapping, Sequence

import numpy as np
import paddle
from paddle.io import DataLoader as PaddleDataLoader, Dataset
from paddle.io import get_worker_info




class DataLoader(PaddleDataLoader):
    def __init__(self, *args, pin_memory=False, **kwargs):
        if "use_shared_memory" not in kwargs:
            kwargs["use_shared_memory"] = bool(pin_memory)
        super().__init__(*args, **kwargs)
        self.pin_memory = bool(pin_memory)

def _is_namedtuple_instance(value):
    return isinstance(value, tuple) and hasattr(value, "_fields")


def default_collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if paddle.is_tensor(elem):
        return paddle.stack(batch, axis=0)
    if isinstance(elem, np.ndarray):
        return paddle.stack([paddle.to_tensor(item) for item in batch], axis=0)
    if isinstance(elem, (float, np.floating)):
        return paddle.to_tensor(batch, dtype=paddle.float32)
    if isinstance(elem, (int, np.integer, bool, np.bool_)):
        return paddle.to_tensor(batch)
    if isinstance(elem, str):
        return list(batch)
    if isinstance(elem, Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    if _is_namedtuple_instance(elem):
        return type(elem)(*(default_collate(items) for items in zip(*batch)))
    if isinstance(elem, tuple):
        return tuple(default_collate(items) for items in zip(*batch))
    if isinstance(elem, Sequence):
        return [default_collate(items) for items in zip(*batch)]
    return batch


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class Subset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return math.ceil(len(self.sampler) / self.batch_size)
