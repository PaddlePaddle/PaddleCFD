import math

import numpy as np

from kappadata.utils.distributed import get_rank, get_world_size


class DistributedSampler:
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False, num_repeats=1):
        self.dataset = dataset
        self.num_replicas = num_replicas or get_world_size()
        self.rank = rank or get_rank()
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_repeats = num_repeats
        self.epoch = 0

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    @property
    def effective_length(self):
        return len(self.dataset)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            indices = rng.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        if self.num_repeats > 1:
            indices = np.repeat(indices, self.num_repeats)[:len(self.dataset)]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices = np.concatenate([indices, np.resize(indices, padding_size)])
        else:
            indices = indices[:self.total_size]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        yield from indices.tolist()
