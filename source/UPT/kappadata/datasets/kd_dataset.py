import logging
from functools import partial

from paddle.io import Dataset

from kappadata.error_messages import getshape_instead_of_getdim
from kappadata.errors import UseModeWrapperException
from kappadata.utils.random import get_rng_from_global


class KDDataset(Dataset):
    def __init__(self, collators=None):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self._collators = collators

        # getdim_... is an alias -> should be defined via getshape_...
        getdim_names = [name for name in dir(self) if name.startswith("getdim_")]
        assert len(getdim_names) == 0, getshape_instead_of_getdim(getdim_names)

    def __getattr__(self, item):
        # make getdim_... an alias to getshape_...[0]
        if item.startswith("getdim_"):
            return partial(self.getdim, item[len("getdim_"):])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __len__(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()

    def dispose(self):
        """ release resources occupied by dataset (e.g. filehandles) """
        pass

    @property
    def collators(self):
        return self._collators or []

    @property
    def root_dataset(self):
        return self

    @property
    def fused_operations(self):
        return []

    @property
    def requires_propagate_ctx(self):
        return False

    @staticmethod
    def has_wrapper(wrapper):
        return False

    @staticmethod
    def has_wrapper_type(wrapper_type):
        return False

    @property
    def all_wrappers(self):
        return []

    @property
    def all_wrapper_types(self):
        return []

    def get_wrapper_of_type(self, wrapper_type):
        wrappers = self.get_wrappers_of_type(wrapper_type)
        if len(wrappers) == 0:
            return None
        assert len(wrappers) == 1
        return wrappers[0]

    def get_wrappers_of_type(self, wrapper_type):
        return []

    def worker_init_fn(self, rank, **kwargs):
        if self.collators is not None:
            rng = get_rng_from_global()
            for collator in self.collators:
                collator.set_rng(rng)

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def getshape(self, kind):
        attr = f"getshape_{kind}"
        assert hasattr(self, attr)
        return getattr(self, attr)()

    def getdim(self, kind):
        shape = self.getshape(kind)
        assert isinstance(shape, tuple) and len(shape) == 1
        return shape[0]
