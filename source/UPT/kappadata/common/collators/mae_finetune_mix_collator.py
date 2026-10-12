import kappadata.collators as kdc


class MAEFinetuneMixCollator(kdc.KDComposeCollator):
    def __init__(self):
        mix_collator = kdc.KDMixCollator(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            mixup_p=0.5,
            cutmix_p=0.5,
            apply_mode="batch",
            lamb_mode="batch",
            shuffle_mode="flip",
        )
        super().__init__(
            collators=[mix_collator],
            dataset_mode="x class",
        )
