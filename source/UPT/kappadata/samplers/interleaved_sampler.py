import bisect
from dataclasses import dataclass

from kappadata.compat.data import ConcatDataset
from kappadata.compat.data import default_collate, DataLoader


@dataclass
class InterleavedSamplerConfig:
    sampler: object
    every_n_epochs: int = None
    every_n_updates: int = None
    every_n_samples: int = None
    collator: callable = None
    batch_size: int = None

    def __str__(self):
        strs = []
        if self.every_n_epochs is not None:
            strs.append(f"every_n_epochs={self.every_n_epochs}")
        if self.every_n_updates is not None:
            strs.append(f"every_n_updates={self.every_n_updates}")
        if self.every_n_samples is not None:
            strs.append(f"every_n_samples={self.every_n_samples}")
        if self.batch_size is not None:
            strs.append(f"batch_size={self.batch_size}")
        return f"{type(self).__name__}({','.join(strs)})"


# can't be a local class as it is required to be pickleable
# AttributeError: Can't pickle local object 'InterleavedSampler.__init__.<locals>._InterleavedConcatDataset'
class _InterleavedConcatDataset(ConcatDataset):
    """ same as ConcatDataset but it returns the dataset index """

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, self.datasets[dataset_idx][sample_idx]

    def worker_init_fn(self, rank, **kwargs):
        for dataset in self.datasets:
            dataset.worker_init_fn(rank, dataset_len=len(dataset), **kwargs)


# can't be a local class as it is required to be pickleable
# AttributeError: Can't pickle local object 'InterleavedSampler.__init__.<locals>._InterleavedCollator'
class _InterleavedCollator:
    def __init__(self, collators):
        self.collators = collators

    def __call__(self, data):
        dataset_idxs, data = zip(*data)
        assert all(dataset_idxs[0] == idx for idx in dataset_idxs)
        return self.collators[dataset_idxs[0]](data)


# can't be a local class as it is required to be pickleable
# AttributeError: Can't pickle local object 'InterleavedSampler.__init__.<locals>._InterleavedBatchSampler'
class _InterleavedBatchSampler:
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def __iter__(self):
        idxs = []
        for is_full_batch, idx in self.sampler:
            idxs.append(idx)
            if is_full_batch:
                yield idxs
                idxs = []
        assert len(idxs) == 0

    def __len__(self):
        raise NotImplementedError


class InterleavedSampler:
    def __init__(
            self,
            main_sampler,
            batch_size,
            configs=None,
            # properties of main sampler
            drop_last=True,
            main_collator=None,
            # duration of InterleavedSampler
            epochs=None,
            updates=None,
            samples=None,
            start_epoch=None,
            start_update=None,
            start_sample=None,
            # batch_size used for dropping last batch
            drop_last_batch_size=None,
    ):
        super().__init__()
        assert isinstance(batch_size, int) and 0 < batch_size
        assert batch_size <= len(main_sampler)
        if drop_last_batch_size is not None:
            assert drop_last and drop_last_batch_size % batch_size == 0
            assert isinstance(drop_last_batch_size, int) and batch_size <= drop_last_batch_size
            assert drop_last_batch_size <= len(main_sampler)
        assert epochs is None or (isinstance(epochs, int) and 0 <= epochs)
        assert updates is None or (isinstance(updates, int) and 0 <= updates)
        assert samples is None or (isinstance(samples, int) and 0 <= samples)
        assert sum([epochs is not None, updates is not None, samples is not None]) == 1
        configs = configs or []
        for config in configs:
            assert (
                    (config.every_n_epochs is not None) or
                    (config.every_n_updates is not None) or
                    (config.every_n_samples is not None)
            )
            assert config.every_n_epochs is None or 0 < config.every_n_epochs
            assert config.every_n_updates is None or 0 < config.every_n_updates
            assert config.every_n_samples is None or 0 < config.every_n_samples
            assert config.batch_size is None or 0 < config.batch_size

        # infer full start checkpoint from one of epoch/update/sample
        if start_epoch is not None:
            assert isinstance(start_epoch, int) and start_update is None and start_sample is None
            start_update = len(main_sampler) // batch_size * start_epoch
            start_sample = start_update * batch_size
        elif start_update is not None:
            assert start_epoch is None and isinstance(start_update, int) and start_sample is None
            start_epoch = int(start_update / (len(main_sampler) // batch_size))
            start_sample = start_update * batch_size
            if start_update % (len(main_sampler) // batch_size) != 0 or not drop_last:
                raise NotImplementedError("defining start_update would require to skip forward in the sampler")
        elif start_sample is not None:
            assert start_epoch is None and start_update is None and isinstance(start_sample, int)
            assert start_sample % batch_size == 0
            start_update = start_sample // batch_size
            start_epoch = int(start_update / (len(main_sampler) // batch_size))
            if start_update % (len(main_sampler) // batch_size) != 0 or not drop_last:
                raise NotImplementedError("defining start_update would require to skip forward in the sampler")
        else:
            start_epoch = start_update = start_sample = 0

        self.main_sampler = main_sampler
        self.drop_last = drop_last
        self.configs = configs
        self.batch_size = batch_size
        self.epochs = epochs
        self.updates = updates
        self.samples = samples
        self.start_epoch = start_epoch
        self.start_update = start_update
        self.start_sample = start_sample
        self.drop_last_batch_size = drop_last_batch_size

        def _get_data_source(sampler):
            if hasattr(sampler, "data_source"):
                return sampler.data_source
            if hasattr(sampler, "dataset"):
                return sampler.dataset
            raise NotImplementedError

        self.index_offsets = [len(_get_data_source(self.main_sampler))]
        for config in self.configs[:-1]:
            self.index_offsets.append(self.index_offsets[-1] + len(_get_data_source(config.sampler)))

        self.dataset = _InterleavedConcatDataset(
            [_get_data_source(self.main_sampler)] +
            [_get_data_source(config.sampler) for config in self.configs]
        )

        self.collator = _InterleavedCollator(
            [main_collator or default_collate] +
            [config.collator or default_collate for config in self.configs]
        )

        self.batch_sampler = _InterleavedBatchSampler(self)

    def get_data_loader(
            self,
            num_workers: int = 0,
            pin_memory: bool = False,
            prefetch_factor: int = None,
    ) -> DataLoader:
        # the default value of prefetch_factor changed from 2 to None in pytorch 2.0 -> pass via optional kwarg
        kwargs = {}
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(
            dataset=self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=self.dataset.worker_init_fn,
            **kwargs,
        )

    def __iter__(self):
        if self.epochs == 0 or self.updates == 0 or self.samples == 0:
            assert self.start_epoch == 0 and self.start_update == 0 and self.start_sample == 0
            yield from self._eval_loop()
        else:
            yield from self._training_loop()

    def _eval_loop(self):
        for config_idx, config in enumerate(self.configs):
            index_offset = self.index_offsets[config_idx]
            sample_in_interleaved = 0
            interleaved_batch_size = config.batch_size or self.batch_size
            for interleaved_idx in config.sampler:
                sample_in_interleaved += 1
                if (
                        sample_in_interleaved % interleaved_batch_size == 0 or
                        sample_in_interleaved == len(config.sampler)
                ):
                    yield True, index_offset + interleaved_idx
                else:
                    yield False, index_offset + interleaved_idx

    def _training_loop(self):
        if self.drop_last:
            if self.drop_last_batch_size is not None:
                if len(self.main_sampler) < self.drop_last_batch_size:
                    factor = self.drop_last_batch_size // self.batch_size
                    self.drop_last_batch_size = len(self.main_sampler) - len(self.main_sampler) % factor
                    self.batch_size = self.drop_last_batch_size // factor
                batch_size = self.drop_last_batch_size
            else:
                if len(self.main_sampler) < self.batch_size:
                    self.batch_size = len(self.main_sampler)
                batch_size = self.batch_size
            samples_per_epoch = len(self.main_sampler) // batch_size * batch_size
        else:
            samples_per_epoch = len(self.main_sampler)

        epoch = self.start_epoch
        update = self.start_update
        sample = self.start_sample
        sample_in_update = 0
        sample_at_last_update = 0
        while True:
            sample_in_epoch = 0
            if hasattr(self.main_sampler, "set_epoch"):
                self.main_sampler.set_epoch(epoch)
            for main_idx in self.main_sampler:
                sample += 1
                sample_in_epoch += 1
                sample_in_update += 1
                if sample_in_update == self.batch_size or sample_in_epoch == samples_per_epoch:
                    yield True, main_idx
                else:
                    yield False, main_idx
                # check if interleaved dataset has to be iterated (only possible after a update)
                # sample_in_update == self.batch_size -> full batch
                # if not drop_last -> last batch is not full but is also an update
                if sample_in_update == self.batch_size or sample_in_epoch == samples_per_epoch:
                    # keep track of what the sample counter was at the last update for every_n_sample checks
                    sample_in_update = 0
                    # increase counters
                    update += 1
                    if sample_in_epoch == samples_per_epoch:
                        epoch += 1

                    for config_idx, config in enumerate(self.configs):
                        # check if interleaved dataset has to be iterated
                        should_iter = False
                        if config.every_n_epochs is not None:
                            # can only occour at the end of an epoch
                            should_iter = sample_in_epoch == samples_per_epoch and epoch % config.every_n_epochs == 0
                        if config.every_n_updates is not None:
                            should_iter = update % config.every_n_updates == 0
                        if config.every_n_samples is not None:
                            if sample % config.every_n_samples == 0:
                                should_iter = True
                            elif sample_at_last_update // config.every_n_samples < sample // config.every_n_samples:
                                should_iter = True
                        if not should_iter:
                            continue
                        index_offset = self.index_offsets[config_idx]
                        interleaved_batch_size = config.batch_size or self.batch_size
                        sample_in_interleaved = 0
                        for interleaved_idx in config.sampler:
                            sample_in_interleaved += 1
                            if (
                                    sample_in_interleaved % interleaved_batch_size == 0 or
                                    sample_in_interleaved == len(config.sampler)
                            ):
                                # yield True, index_offset + interleaved_idx
                                if isinstance(interleaved_idx, list):
                                    for idx in interleaved_idx:
                                        yield True, index_offset + idx
                                else:
                                    yield True, index_offset + interleaved_idx
                            else:
                                # yield False, index_offset + interleaved_idx
                                if isinstance(interleaved_idx, list):
                                    # 如果返回的是列表，循环处理每一个索引
                                    for idx in interleaved_idx:
                                        yield False, index_offset + idx
                                else:
                                    # 如果是单个整数，按原逻辑处理
                                    yield False, index_offset + interleaved_idx
                    sample_at_last_update = sample
                    # check if end is reached
                    if (
                            (self.epochs is not None and epoch == self.epochs) or
                            (self.updates is not None and update == self.updates) or
                            (self.samples is not None and sample >= self.samples)
                    ):
                        return
                    # if drop_last -> skip last non-full batch
                    if sample_in_epoch == samples_per_epoch:
                        break
