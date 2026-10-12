from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.one_hot import to_one_hot_vector


class OneHotWrapper(KDWrapper):
    def getitem_class(self, idx, ctx=None):
        y = self.dataset.getitem_class(idx, ctx)
        return to_one_hot_vector(y, n_classes=self.dataset.getdim_class())
