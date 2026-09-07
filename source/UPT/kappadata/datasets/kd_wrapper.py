from functools import partial

from kappadata.errors import UseModeWrapperException
from .kd_dataset import KDDataset


class KDWrapper(KDDataset):
    def __init__(self, dataset: KDDataset, ctx_prefix: str = None):
        super().__init__()
        self.dataset = dataset
        self.ctx_prefix = ctx_prefix or type(self).__name__
        # children should overwrite _worker_init_fn
        assert type(self).worker_init_fn == KDWrapper.worker_init_fn

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        # make getdim_... an alias to getshape_...[0]
        if item.startswith("getdim_"):
            return partial(self.getdim, item[len("getdim_"):])
        return getattr(self.dataset, item)

    def getshape(self, kind):
        attr = f"getshape_{kind}"
        assert hasattr(self, attr)
        return getattr(self, attr)()

    def getdim(self, kind):
        shape = self.getshape(kind)
        assert isinstance(shape, tuple) and len(shape) == 1
        return shape[0]

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def dispose(self):
        self.dataset.dispose()

    @property
    def collators(self):
        assert self._collators is None, "register collators on root datset"
        return self.dataset.collators

    @property
    def root_dataset(self):
        # KDDataset implements root_dataset -> __getitem__ doesn't trigger
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

    def get_wrappers_of_type(self, wrapper_type):
        wrappers = self.dataset.get_wrappers_of_type(wrapper_type)
        if type(self) == wrapper_type:
            return [self] + wrappers
        return wrappers

    def worker_init_fn(self, rank, **kwargs):
        self._worker_init_fn(rank, **kwargs)
        self.dataset.worker_init_fn(rank, **kwargs)

    def _worker_init_fn(self, rank, **kwargs):
        pass
