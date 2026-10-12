from .kd_collator_base import KDCollatorBase
from .kd_single_collator import KDSingleCollator


class KDComposeCollator(KDCollatorBase):
    def __init__(self, collators, dataset_mode: str, return_ctx: bool = False):
        super().__init__(dataset_mode=dataset_mode, return_ctx=return_ctx)
        assert self.dataset_mode is not None, "KDComposeCollator requires dataset_mode"
        assert self.return_ctx is not None, "KDComposeCollator requires return_ctx"

        assert isinstance(collators, list) and len(collators) > 0
        assert all(isinstance(c, KDSingleCollator) for c in collators)
        self.collators = collators
        assert all(c.default_collate_mode in [None, "before", "after"] for c in collators)

    def set_rng(self, rng):
        for collator in self.collators:
            collator.set_rng(rng)
        return self

    def __call__(self, batch):
        return self._call_impl(
            batch=batch,
            collators=self.collators,
            dataset_mode=self.dataset_mode,
            return_ctx=self.return_ctx,
        )
