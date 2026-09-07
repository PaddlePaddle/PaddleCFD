from kappadata.utils.distributed import get_rank, get_world_size


class SamplerBase:
    def __init__(self, rank=None, world_size=None):
        super().__init__()
        self.rank = rank or get_rank()
        self.world_size = world_size or get_world_size()
        self.epoch = 0

    @property
    def effective_length(self):
        raise NotImplementedError

    def __len__(self):
        # adjust to length-per-device and cutoff trailing samples for distributed
        return self.effective_length // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _generate_indices(self):
        raise NotImplementedError

    def __iter__(self):
        indices = self._generate_indices()
        # distribute among ranks
        indices = indices[self.rank:self.total_size:self.world_size]
        # drop last
        indices = indices[:len(self)]
        yield from indices