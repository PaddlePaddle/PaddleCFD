import numpy as np

from kappadata.utils.distributed import get_rank, get_world_size
from kappadata.utils.getall_as_tensor import getall_as_tensor


class SemiSampler:
    def __init__(self, dataset, num_labeled=1, num_unlabeled=1, rank=None, world_size=None, seed=0, length_mode='unlabeled'):
        super().__init__()
        assert 1 <= num_labeled
        assert 1 <= num_unlabeled
        self.dataset = dataset
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0
        self.seed = seed
        assert length_mode in ['labeled', 'unlabeled', 'all']
        self.length_mode = length_mode

        self.classes = getall_as_tensor(dataset)
        is_unlabeled = self.classes == -1
        self.labeled_idxs = (~is_unlabeled).nonzero().flatten().tolist()
        self.unlabeled_idxs = is_unlabeled.nonzero().flatten().tolist()
        assert len(self.labeled_idxs) > 0 and len(self.unlabeled_idxs) > 0

    @property
    def effective_length(self):
        if self.length_mode == 'labeled':
            num_chunks = len(self.labeled_idxs) // self.num_labeled
        elif self.length_mode == 'unlabeled':
            num_chunks = len(self.unlabeled_idxs) // self.num_unlabeled
        else:
            num_chunks = (len(self.labeled_idxs) + len(self.unlabeled_idxs)) // (self.num_labeled + self.num_unlabeled)
        return num_chunks * (self.num_labeled + self.num_unlabeled)

    def __len__(self):
        return self.effective_length // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.rank * 9973 + self.epoch * 433)

        def iterator(idxs):
            while True:
                yield from rng.permutation(len(idxs)).tolist()

        labeled_iterator = iterator(self.labeled_idxs)
        unlabeled_iterator = iterator(self.unlabeled_idxs)
        for i in range(len(self)):
            if i % (self.num_labeled + self.num_unlabeled) < self.num_labeled:
                yield self.labeled_idxs[next(labeled_iterator)]
            else:
                yield self.unlabeled_idxs[next(unlabeled_iterator)]
