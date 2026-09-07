import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.factory import object_to_transform
from kappadata.transforms import KDTransform
from kappadata.transforms.semseg import (
    KDSemsegPad,
    KDSemsegRandomResize,
    KDSemsegRandomHorizontalFlip,
    KDSemsegRandomCrop,
    KDSemsegResize,
)


class SemsegTransformWrapper(KDWrapper):
    _SEMSEG_TRANSFORMS = (
        KDSemsegPad,
        KDSemsegRandomResize,
        KDSemsegRandomHorizontalFlip,
        KDSemsegRandomCrop,
        KDSemsegResize,
    )

    def __init__(self, dataset, transforms, seed=None):
        super().__init__(dataset=dataset)
        self.transforms = [object_to_transform(transform) for transform in transforms]
        self.seed = seed

    @property
    def fused_operations(self):
        return super().fused_operations + [["x", "semseg"]]

    def getitem_x(self, idx, ctx=None):
        return self.getitem_xsemseg(idx, ctx=ctx)[0]

    def getitem_semseg(self, idx, ctx=None):
        return self.getitem_xsemseg(idx, ctx=ctx)[1]

    def getitem_xsemseg(self, idx, ctx=None):
        x = self.dataset.getitem_x(idx, ctx=ctx)
        semseg = self.dataset.getitem_semseg(idx, ctx=ctx)
        if self.seed is not None:
            rng = np.random.default_rng(seed=self.seed + idx)
        else:
            rng = None
        for transform in self.transforms:
            if isinstance(transform, self._SEMSEG_TRANSFORMS):
                if rng is not None:
                    transform.set_rng(rng)
                x, semseg = transform((x, semseg), ctx=ctx)
            else:
                if rng is not None and isinstance(transform, KDTransform):
                    transform.set_rng(rng)
                x = transform(x, ctx=ctx)
        return x, semseg
