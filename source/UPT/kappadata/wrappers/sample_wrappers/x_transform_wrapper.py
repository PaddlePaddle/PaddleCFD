from .base.transform_wrapper_base import TransformWrapperBase


class XTransformWrapper(TransformWrapperBase):
    def getitem_x(self, idx, ctx=None):
        item = self.dataset.getitem_x(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx)

    def getitem_class(self, idx, ctx=None):
        # TODO ugly solution to circumvent XTransformWrapper being skipped when a dataset with a fused operation
        #  is wrapperd (e.g. XTransformWrapper(KDMixWrapper(dataset)) -> this method is only here because
        #  operations of a fused operation are checked to be available in the wrapper type
        return self.dataset.getitem_class(idx, ctx=ctx)

    def getitem_xclass(self, idx, ctx=None):
        # TODO ugly solution to circumvent XTransformWrapper being skipped when a dataset with a fused operation
        #  is wrapperd (e.g. XTransformWrapper(KDMixWrapper(dataset))
        item, cls = self.dataset.getitem_xclass(idx, ctx=ctx)
        return self._getitem(item=item, idx=idx, ctx=ctx), cls