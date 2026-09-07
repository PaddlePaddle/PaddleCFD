from kappadata.utils.random import get_rng_from_global
from .kd_collator_base import KDCollatorBase


class KDSingleCollator(KDCollatorBase):
    def __init__(self, dataset_mode: str = None, return_ctx: bool = None):
        # dataset_mode/return_ctx is only needed when KDCollator is called directly (i.e. not via KDComposeCollator)
        super().__init__(dataset_mode=dataset_mode, return_ctx=return_ctx)
        self.rng = get_rng_from_global()

    def set_rng(self, rng):
        self.rng = rng
        return self

    def __call__(self, batch):
        assert self.dataset_mode is not None and self.return_ctx is not None, \
            "use KDCollator as part of KDComposeCollator or specify dataset_mode and return_ctx"
        return self._call_impl(
            batch=batch,
            collators=[self],
            dataset_mode=self.dataset_mode,
            return_ctx=self.return_ctx,
        )

    @property
    def default_collate_mode(self):
        raise NotImplementedError

    def collate(self, batch, dataset_mode, ctx=None):
        raise NotImplementedError
