from kappadata.common.transforms.imagenet_minaug_transforms import ImagenetMinaugTransform
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper


class ImagenetMinaugXTransformWrapper(XTransformWrapper):
    def __init__(self, dataset, size=224, interpolation="bicubic", min_scale=0.08, **kwargs):
        super().__init__(
            dataset,
            transform=ImagenetMinaugTransform(
                size=size,
                interpolation=interpolation,
                min_scale=min_scale,
            ),
            **kwargs,
        )
