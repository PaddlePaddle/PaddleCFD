from kappadata.compat.data import default_collate

from kappadata.utils.random import get_rng_from_global


class KDCollatorBase:
    def __init__(self, dataset_mode: str, return_ctx: bool):
        self.dataset_mode = dataset_mode
        self.return_ctx = return_ctx

    def set_rng(self, rng):
        raise NotImplementedError

    def worker_init_fn(self, *_, **__):
        self.set_rng(get_rng_from_global())

    def __call__(self, batch):
        raise NotImplementedError

    @staticmethod
    def _call_impl(batch, collators, dataset_mode, return_ctx):
        called_default_collate = False
        removed_ctx_from_batch = False
        ctx = {}
        for collator in collators:
            if collator.default_collate_mode is None:
                assert not called_default_collate

            # check if default_collate should be called before this collator
            if collator.default_collate_mode == "before" and not called_default_collate:
                batch = default_collate(batch)
                if return_ctx:
                    batch, ctx = batch
                    assert isinstance(ctx, dict), \
                        "ModeWrapper.return_ctx should be equal to KDComposeCollator.return_ctx"
                called_default_collate = True

            # collate ctx if not collated already
            if not called_default_collate and return_ctx and not removed_ctx_from_batch:
                batch, ctx = zip(*batch)
                ctx = default_collate(ctx)
                assert isinstance(ctx, dict), "ModeWrapper.return_ctx should be equal to KDComposeCollator.return_ctx"
                removed_ctx_from_batch = True

            # call collator
            batch = collator.collate(batch, dataset_mode, ctx)

            # check if default_collate should be called before this collator
            if collator.default_collate_mode == "after":
                assert not called_default_collate
                batch = default_collate(batch)

        if return_ctx:
            return batch, ctx
        return batch
