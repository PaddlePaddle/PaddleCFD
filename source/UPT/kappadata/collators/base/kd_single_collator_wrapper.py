from .kd_collator_base import KDCollatorBase
from .kd_single_collator import KDSingleCollator


class KDSingleCollatorWrapper(KDCollatorBase):
    def __init__(self, collator, dataset_mode: str, return_ctx: bool = False):
        super().__init__(dataset_mode=dataset_mode, return_ctx=return_ctx)
        assert self.dataset_mode is not None, "KDSingleCollatorWrapper requires dataset_mode"
        assert self.return_ctx is not None, "KDSingleCollatorWrapper requires return_ctx"
        assert isinstance(collator, KDSingleCollator)
        self.collator = collator

    def set_rng(self, rng):
        self.collator.set_rng(rng)
        return self

    def __call__(self, batch):
        batch, ctx = self.collator.collate(batch=batch, dataset_mode=self.dataset_mode, ctx={})
        if self.return_ctx:
            return batch, ctx
        return batch
