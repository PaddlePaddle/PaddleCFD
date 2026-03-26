from .base.transform_wrapper_base import TransformWrapperBase


class SourceTransformWrapper(TransformWrapperBase):
    def getitem_source(self, idx, ctx=None):
        item = self.dataset.getitem_source(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)
