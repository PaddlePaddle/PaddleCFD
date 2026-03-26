from paddle.vision.datasets import ImageFolder
from paddle.vision.datasets.folder import default_loader

from kappadata.datasets.kd_dataset import KDDataset


class KDImageFolder(KDDataset):
    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
    ):
        super().__init__()
        self.dataset = ImageFolder(
            root=root,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.transform = transform

    # noinspection PyUnusedLocal
    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        x = self.transform(x, ctx=ctx)
        return x

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return self.dataset.targets[idx]

    def getshape_class(self):
        return len(self.dataset.classes),

    def __len__(self):
        return len(self.dataset)
