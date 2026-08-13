from typing import Iterator

import numpy as np
import paddle

__all__ = ["MultisetSampler"]


class MultisetSampler(paddle.io.Sampler):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: paddle.io.Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        rank=0,
        world_size=1,
        distributed=True,
    ) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed

    def _build_subdataset_indices(self):
        """Build per-rank indices for each sub-dataset."""
        subdataset_indices = []
        for subdataset_idx, subdataset in enumerate(self.sub_dsets):
            indices = np.arange(len(subdataset), dtype=np.int64)
            if self.shuffle:
                rng = np.random.default_rng(
                    1000 * self.epoch + 100 * self.seed + 10 * self.rank + subdataset_idx
                )
                rng.shuffle(indices)
            if self.distributed and self.world_size > 1:
                indices = indices[self.rank :: self.world_size]
            usable_size = (len(indices) // self.batch_size) * self.batch_size
            subdataset_indices.append(indices[:usable_size].tolist())
        return subdataset_indices

    def __iter__(self) -> Iterator[int]:
        subdataset_indices = self._build_subdataset_indices()
        samplers = [iter(indices) for indices in subdataset_indices]
        sampler_choices = [
            idx for idx, indices in enumerate(subdataset_indices) if len(indices) > 0
        ]
        rng = np.random.default_rng(100 * self.epoch + 10 * self.seed + self.rank)
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = int(rng.integers(low=0, high=len(sampler_choices)))
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            try:
                queue = []
                for i in range(self.batch_size):
                    queue.append(next(samplers[dset_sampled]) + offset)
                if len(queue) == self.batch_size:
                    for d in queue:
                        yield d
            except Exception as err:
                print("ERRRR", err)
                sampler_choices.pop(index_sampled)
                print(
                    f"Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}"
                )
                continue
            if count >= self.max_samples:
                break

    def __len__(self) -> int:
        available_batches = sum(
            len(indices) // self.batch_size for indices in self._build_subdataset_indices()
        )
        return min(self.max_samples, available_batches) * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
