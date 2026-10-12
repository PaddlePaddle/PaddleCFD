from kappadata.common.transforms.imagenet_noaug_transforms import ImagenetNoaugTransform
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper


class ImagenetNoaugXTransformWrapper(XTransformWrapper):
    def __init__(self, dataset, resize_size=256, center_crop_size=224, interpolation="bicubic", **kwargs):
        super().__init__(
            dataset,
            transform=ImagenetNoaugTransform(
                resize_size=resize_size,
                center_crop_size=center_crop_size,
                interpolation=interpolation,
            ),
            **kwargs,
        )
