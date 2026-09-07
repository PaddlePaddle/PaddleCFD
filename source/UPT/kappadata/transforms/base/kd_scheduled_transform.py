import paddle
from kappaschedules import LinearIncreasingSchedule, object_to_schedule

from kappadata.factory import object_to_transform
from .kd_transform import KDTransform


class KDScheduledTransform(KDTransform):
    def __init__(self, transform, schedule=None):
        super().__init__()
        self.transform = object_to_transform(transform)
        self.rank = None
        self.num_workers = None
        self.batch_size = None
        self.n_batches = None
        self.sample_counter = 0
        self.ctx_key = f"{self.ctx_prefix}.strength"
        # default to linear from [0, 1]
        self.schedule = object_to_schedule(schedule) or LinearIncreasingSchedule()

    def _worker_init_fn(
            self,
            rank,
            num_workers,
            batch_size=None,
            dataset_len=None,
            world_size=None,
            drop_last=None,
            epochs=None,
            updates=None,
            samples=None,
            **__,
    ):
        self.rank = rank
        self.num_workers = num_workers
        assert batch_size is not None
        self.batch_size = batch_size

        # calculate total_n_batches (number of batches independent of num_workers)
        if epochs is not None:
            assert updates is None and samples is None
            assert drop_last is not None
            assert dataset_len is not None
            assert world_size is not None
            assert isinstance(epochs, int) and epochs >= 0
            dataset_len //= world_size
            if drop_last:
                batches_per_epoch = dataset_len // batch_size
            else:
                batches_per_epoch = (dataset_len + batch_size - 1) // batch_size
            self.n_batches = epochs * batches_per_epoch
        elif updates is not None:
            assert samples is None
            assert isinstance(updates, int) and updates >= 0
            self.n_batches = updates
        elif samples is not None:
            assert isinstance(samples, int) and samples >= 0
            if samples % batch_size == 0:
                self.n_batches = samples // batch_size
            else:
                self.n_batches = samples // batch_size + 1
        else:
            raise NotImplementedError

    def __call__(self, x, ctx=None):
        # make sure that worker_init_fn was called
        if get_worker_info() is not None:
            assert self.n_batches is not None, "call KDScheduledTransform.worker_init_fn before applying the transform"
        # scale_strength when called in worker process
        if self.n_batches is not None:
            # caulculate progress
            batch_idx = self.sample_counter // self.batch_size * self.num_workers + self.rank
            strength = self.schedule.get_value(batch_idx, self.n_batches)
            self.sample_counter += 1

            # scale
            self.transform.scale_strength(strength)

            if ctx is not None:
                ctx[self.ctx_key] = strength
        return self.transform(x, ctx=ctx)
