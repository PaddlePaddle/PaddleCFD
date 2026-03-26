from .base.transform_wrapper_base import TransformWrapperBase


class YTransformWrapper(TransformWrapperBase):
    def getitem_y(self, idx, ctx=None):
        item = self.dataset.getitem_y(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)
