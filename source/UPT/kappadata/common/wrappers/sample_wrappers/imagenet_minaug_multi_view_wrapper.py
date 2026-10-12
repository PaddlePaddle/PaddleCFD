from kappadata.common.transforms.imagenet_minaug_transforms import ImagenetMinaugTransform
from kappadata.wrappers.sample_wrappers.kd_multi_view_wrapper import KDMultiViewWrapper, KDMultiViewConfig


class ImagenetMinaugMultiViewWrapper(KDMultiViewWrapper):
    def __init__(self, dataset, n_views=2, size=224, interpolation="bicubic", min_scale=0.08, **kwargs):
        super().__init__(
            dataset=dataset,
            configs=[
                KDMultiViewConfig(
                    n_views=n_views,
                    transform=ImagenetMinaugTransform(
                        size=size,
                        interpolation=interpolation,
                        min_scale=min_scale,
                    ),
                ),
            ],
            **kwargs
        )
