from .base.transform_wrapper_base import TransformWrapperBase


class TargetTransformWrapper(TransformWrapperBase):
    def getitem_target(self, idx, ctx=None):
        item = self.dataset.getitem_target(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)
