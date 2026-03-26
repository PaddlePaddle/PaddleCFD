import numpy as np
import paddle

from kappadata.utils.distributed import get_rank, get_world_size


class WeightedSampler:
    def __init__(self, dataset, weights, size=None, seed=0, rank=None, world_size=None):
        super().__init__()
        assert len(dataset) == len(weights)
        self.dataset = dataset
        self.weights = weights.numpy() if paddle.is_tensor(weights) else np.asarray(weights)
        self.size = size
        self.seed = seed
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0

    @property
    def effective_length(self):
        if self.size is None:
            return len(self.dataset)
        assert len(self.dataset) >= self.size, f"{len(self.dataset)} < {self.size}"
        return self.size

    def __len__(self):
        return self.effective_length // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        probs = self.weights / self.weights.sum()
        indices = rng.choice(len(self.dataset), size=self.effective_length, replace=False, p=probs)
        indices = indices[self.rank:self.effective_length:self.world_size]
        yield from indices[:len(self)].tolist()
