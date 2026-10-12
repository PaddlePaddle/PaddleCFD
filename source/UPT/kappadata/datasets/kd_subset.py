from functools import partial

import paddle
from kappadata.compat.data import Subset

from kappadata.errors import UseModeWrapperException


class KDSubset(Subset):
    def __getattr__(self, item):
        if item.startswith("getitem_"):
            # all methods starting with getitem_ are called with self.indices[idx]
            func = getattr(self.dataset, item)
            return partial(self._call_getitem, func)
        if item == "dataset":
            return getattr(super(), item)
        if item.startswith("getall_"):
            # subsample getitem_ with the indices
            return partial(self._call_getall, item)
        return getattr(self.dataset, item)

    def _call_getitem(self, func, idx, *args, **kwargs):
        return func(self.indices[idx], *args, **kwargs)

    def _call_getall(self, item):
        result = getattr(self.dataset, item)()
        return [result[i] for i in self.indices]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    @property
    def collators(self):
        return self.dataset.collators

    @property
    def root_dataset(self):
        return self.dataset.root_dataset

    @property
    def fused_operations(self):
        return self.dataset.fused_operations

    @property
    def requires_propagate_ctx(self):
        return self.dataset.requires_propagate_ctx

    def has_wrapper(self, wrapper):
        if self == wrapper:
            return True
        return self.dataset.has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if type(self) == wrapper_type:
            return True
        return self.dataset.has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        return [self] + self.dataset.all_wrappers

    @property
    def all_wrapper_types(self):
        return [type(self)] + self.dataset.all_wrapper_types

    def get_wrapper_of_type(self, wrapper_type):
        wrappers = self.get_wrappers_of_type(wrapper_type)
        if len(wrappers) == 0:
            return None
        assert len(wrappers) == 1
        return wrappers[0]

    def get_wrappers_of_type(self, wrapper_type):
        wrappers = self.dataset.get_wrappers_of_type(wrapper_type)
        if type(self) == wrapper_type:
            return [self] + wrappers
        return wrappers

    def worker_init_fn(self, rank, **kwargs):
        self.dataset.worker_init_fn(rank, **kwargs)

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def get_sampler_weights(self):
        sampler_weights = self.dataset.get_sampler_weights()
        assert paddle.is_tensor(sampler_weights)
        return sampler_weights[self.indices]
