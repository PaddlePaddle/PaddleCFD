import logging

import kappadata as kd
import paddle
from distributed.config import is_distributed
from kappadata.collators import KDComposeCollator, KDSingleCollatorWrapper
from kappadata.samplers import (ClassBalancedSampler, DistributedSampler,
                                RandomSampler, SemiSampler, SequentialSampler,
                                WeightedSampler)
from kappadata.wrappers import ModeWrapper, ShuffleWrapper, SubsetWrapper
from providers.config_providers.noop_config_provider import NoopConfigProvider
from utils.num_worker_heuristic import get_fair_cpu_count, get_total_cpu_count
from utils.seed import get_random_int


class DataContainer:
    def __init__(
        self,
        num_workers=None,
        max_num_workers=None,
        pin_memory=None,
        prefetch_factor=2,
        config_provider=None,
        seed=None,
        **datasets,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.num_workers = num_workers
        self.max_num_workers = max_num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.config_provider = config_provider or NoopConfigProvider()
        # self.generator = paddle.Generator()
        self.generator = None
        # if seed is not None:
        #     self.generator.seed(seed)
        self.datasets = datasets
        self.persistent_loaders = {}
        self.added_to_config_provider = False
        self.run_type = "train"
        if "train" not in self.datasets:
            self.datasets["train"] = list(self.datasets.values())[0]

    def get_dataset(self, key=None, mode=None, max_size=None, shuffle_seed=None):
        key = key or list(self.datasets.keys())[0]
        dataset = self.datasets[key]
        if shuffle_seed is not None:
            dataset = ShuffleWrapper(dataset=dataset, seed=shuffle_seed)
        if max_size is not None:
            dataset = SubsetWrapper(dataset, end_index=max_size)
        if mode is not None:
            dataset = ModeWrapper(dataset=dataset, mode=mode, return_ctx=True)
            if len(dataset.collators) == 1:
                collator = KDSingleCollatorWrapper(
                    collator=dataset.collators[0],
                    dataset_mode=dataset.mode,
                    return_ctx=dataset.return_ctx,
                )
            elif len(dataset.collators) > 1:
                collator = KDComposeCollator(
                    collators=dataset.collators,
                    dataset_mode=dataset.mode,
                    return_ctx=dataset.return_ctx,
                )
            else:
                collator = None
            return dataset, collator
        return dataset

    def get_main_sampler(
        self,
        train_dataset,
        num_repeats=1,
        shuffle=True,
        num_unlabeled_per_labeled=None,
        weighted_size=None,
        samples_per_class=None,
        getall_item="class",
    ):
        if samples_per_class is not None:
            assert num_repeats == 1
            assert num_unlabeled_per_labeled is None
            assert weighted_size is None
            return ClassBalancedSampler(
                dataset=train_dataset,
                samples_per_class=samples_per_class,
                shuffle=shuffle,
                getall_item=getall_item,
            )
        if weighted_size is not None:
            assert num_repeats == 1 and shuffle and num_unlabeled_per_labeled is None
            self.logger.info(f"main_sampler: WeightedSampler(size={weighted_size})")
            return WeightedSampler(
                dataset=train_dataset,
                weights=train_dataset.get_sampler_weights(),
                size=weighted_size,
            )
        if num_unlabeled_per_labeled is not None:
            assert num_repeats == 1 and shuffle and weighted_size is None
            self.logger.info(
                f"main_sampler: SemiSampler(num_unlabeled_per_labeled={num_unlabeled_per_labeled})"
            )
            seed = get_random_int(generator=self.generator)
            return SemiSampler(
                dataset=train_dataset,
                num_labeled=1,
                num_unlabeled=num_unlabeled_per_labeled,
                seed=seed,
            )
        if is_distributed():
            import paddle.distributed as dist
            seed = get_random_int(generator=self.generator)
            assert num_unlabeled_per_labeled is None and weighted_size is None
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            self.logger.info(
                f"main_sampler: PaddleCompatible DistributedSampler "
                f"(rank={rank}, world_size={world_size}, num_repeats={num_repeats}, shuffle={shuffle})"
            )
            return DistributedSampler(
                train_dataset,
                num_repeats=num_repeats,
                shuffle=shuffle,
                seed=seed,
                drop_last=True,
                num_replicas=world_size,
                rank=rank,
            )
        if shuffle:
            self.logger.info(f"main_sampler: RandomSampler(num_repeats={num_repeats})")
            return RandomSampler(
                train_dataset, num_repeats=num_repeats, generator=self.generator
            )
        else:
            self.logger.info(f"main_sampler: SequentialSampler")
            return SequentialSampler(train_dataset)

    def get_data_loader(
        self,
        main_sampler,
        main_collator,
        batch_size,
        epochs,
        updates,
        samples,
        configs,
        start_epoch=None,
    ):
        sampler = kd.InterleavedSampler(
            main_sampler=main_sampler,
            batch_size=batch_size,
            configs=configs,
            main_collator=main_collator,
            epochs=epochs,
            updates=updates,
            samples=samples,
            start_epoch=start_epoch,
        )
        if self.num_workers is None:
            num_workers = get_fair_cpu_count()
        else:
            num_workers = self.num_workers
        if self.max_num_workers is not None:
            num_workers = min(self.max_num_workers, num_workers)
        num_workers = 0
        pin_memory = True if self.pin_memory is None else self.pin_memory
        self.logger.info(f"Using paddle.io.DataLoader to wrap kappadata sampler")
        loader = sampler.get_data_loader(
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.prefetch_factor,
        )
        # loader = paddle.io.DataLoader(
        #     dataset=sampler.dataset,         # 使用 sampler 内部组合好的数据集
        #     batch_sampler=sampler,           # 使用 sampler 作为批采样器
        #     num_workers=num_workers,
        #     use_shared_memory=pin_memory,    # 对应 paddle 的 pin_memory 逻辑
        #     return_list=True                 # 确保返回格式正确
        # )
        self.logger.info(
            f"created paddle dataloader (batch_size={batch_size} num_workers={num_workers})"
        )
        self.logger.info(f"concatenated dataset properties:")
        for dataset in sampler.dataset.datasets:
            self.logger.info(
                f"- mode='{dataset.mode}' len={len(dataset)} root_dataset={dataset.root_dataset}"
            )
        if not self.added_to_config_provider:
            self.config_provider.update(
                {
                    f"dataloader/num_workers": loader.num_workers,
                    f"dataloader/pin_memory": loader.pin_memory,
                }
            )
            self.added_to_config_provider = True
        return loader

    def dispose(self):
        for dataset in self.datasets.values():
            dataset.dispose()
