from kappadata.common.transforms.byol_transforms import BYOLTransform0, BYOLTransform1
from kappadata.wrappers.sample_wrappers.kd_multi_view_wrapper import KDMultiViewWrapper


class ByolMultiViewWrapper(KDMultiViewWrapper):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset=dataset, configs=[BYOLTransform0(), BYOLTransform1()], **kwargs)
