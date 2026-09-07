import numpy as np
import paddle

from kappadata.utils.distributed import get_rank, get_world_size
from kappadata.utils.getall_as_tensor import getall_as_tensor


class ClassBalancedSampler:
    def __init__(self, dataset, shuffle=True, samples_per_class=None, getall_item='class', seed=0, rank=None, world_size=None):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0

        self.num_classes = max(2, dataset.getdim_class())
        classes = getall_as_tensor(self.dataset, item=getall_item)
        unique, counts = paddle.unique(classes, return_counts=True)
        assert classes.ndim == 1
        assert len(unique) == self.num_classes

        self.indices_per_class = [(classes == i).nonzero().flatten().numpy() for i in range(self.num_classes)]
        self.samples_per_class = samples_per_class or int(counts.max().item())

    @property
    def effective_length(self):
        return self.num_classes * self.samples_per_class

    def __len__(self):
        return self.effective_length // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = []
        for indices_per_class in self.indices_per_class:
            remaining_indices = self.samples_per_class
            while remaining_indices > 0:
                perm = rng.permutation(len(indices_per_class)) if self.shuffle else np.arange(len(indices_per_class))
                perm = perm[:remaining_indices]
                indices.append(indices_per_class[perm])
                remaining_indices -= len(perm)
        indices = np.concatenate(indices)
        if self.shuffle:
            indices = indices[rng.permutation(len(indices))]
        indices = indices[self.rank:self.effective_length:self.world_size]
        yield from indices[:len(self)].tolist()
