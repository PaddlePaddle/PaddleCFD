from pathlib import Path

import paddle

from kappadata.datasets.kd_wrapper import KDWrapper


class OverwriteClassesWrapper(KDWrapper):
    def __init__(self, dataset, uri=None, classes=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        assert (uri is not None) ^ (classes is not None)
        if uri is not None:
            if isinstance(uri, str):
                uri = Path(uri).expanduser()
            if uri.name.endswith(".th") or uri.name.endswith(".pth"):
                classes = paddle.load(uri)
                assert classes.ndim == 1
            else:
                raise NotImplementedError
        assert len(classes) == len(self.dataset)
        self.classes = classes

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        cls = self.classes[idx]
        if paddle.is_tensor(cls):
            cls = cls.item()
        return cls
