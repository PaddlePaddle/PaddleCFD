from pathlib import Path
from typing import Tuple

import numpy as np
import paddle


def load_data(data_dir: Path):
    print(f"Loading data from {data_dir}")
    u = np.load(data_dir / "u.npy")
    v = np.load(data_dir / "v.npy")
    mask = np.load(data_dir / "mask.npy")
    u = np.pad(u, ((0, 0), (1, 1), (1, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (1, 0)), mode="constant", constant_values=0)
    mask = 1 - np.pad(mask, ((1, 1), (1, 0)), mode="constant", constant_values=1)
    u[:, 1:-1, 0] = 0.5
    u[:, 1:-1, -1] = 0.5
    return u, v, mask


class CfdDataset(paddle.io.Dataset):
    """
    Base class for cfd datasets
    """

    def __geitem__(self, idx: int) -> tuple:
        """
        Returns a tuple of (features, labels, mask)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CfdAutoDataset(paddle.io.Dataset):
    """
    Base class for auto-regressive dataset.
    """

    def __getitem__(
        self, index: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Should return a tuple of (input, labels, mask)"""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class KarmanDataset(CfdDataset):
    def __init__(self, data_dir: Path, time_step_size: int = 10):
        self.data_dir = data_dir
        self.time_step_size = time_step_size
        u, v, mask = load_data(data_dir)
        u = paddle.FloatTensor(u)
        v = paddle.FloatTensor(v)
        self.mask = paddle.FloatTensor(mask)
        self.features = paddle.stack([u, v], dim=1)
        self.labels = self.features[time_step_size:]
        self.features = self.features[:-time_step_size]

    def __getitem__(self, idx: int):
        feat = self.features[idx]
        label = self.labels[idx]
        return feat, self.mask, label

    def __len__(self):
        return len(self.features)
