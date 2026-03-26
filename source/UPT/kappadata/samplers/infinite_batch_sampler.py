from kappadata.compat.data import BatchSampler


class InfiniteBatchSampler(BatchSampler):
    """
    BatchSampler that keeps fetching batches across epoch boundaries.
    This allows a DataLoader to prefetch batches from the next epoch.
    This is especially useful for small datasets as the default
    pytorch BatchSampler will always have to wait for the first batch
    of EVERY epoch to be ready.
    In the extreme case (batch_size == len(dataset)) the dataloading
    will be fully synchronous (with BatchSampler), independent of how
    many workers are used.
    With InfiniteBatchSampler this downtime is avoided.
    TODO code example
    """

    def __init__(self, *args, epochs=None, updates=None, samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert epochs is None or (isinstance(epochs, int) and 0 < epochs)
        assert updates is None or (isinstance(updates, int) and 0 < updates)
        assert samples is None or (isinstance(samples, int) and 0 < samples)
        assert sum([epochs is not None, updates is not None, samples is not None]) <= 1
        self.epochs = epochs
        self.updates = updates
        self.samples = samples

    def __iter__(self):
        epochs = 0
        updates = 0
        samples = 0
        while True:
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epochs)
            for batch in super().__iter__():
                updates += 1
                samples += len(batch)
                yield batch
            epochs += 1
            if (
                    (self.epochs is not None and epoch == self.epochs) or
                    (self.updates is not None and update == self.updates) or
                    (self.samples is not None and sample >= self.samples)
            ):
                break

    def __len__(self):
        raise NotImplementedError
