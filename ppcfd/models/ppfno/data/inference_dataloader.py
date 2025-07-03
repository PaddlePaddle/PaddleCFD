import copy
import logging
import os
import random
import re
import sys
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from timeit import default_timer
from typing import Dict, List, Optional, Tuple, Union

import meshio
import numpy as np
import open3d as o3d
import paddle

from .base_datamodule import BaseDataModule

from ..neuralop.utils import UnitGaussianNormalizer


def get_last_dir(path):
    return os.path.basename(os.path.normpath(path))


class LoadFile:

    def __init__(self, path):
        self.path = path

    def index_to_file(self, filename: str, extension: str = ".npy") -> Path:
        return self.path / (filename + extension)

    def load_file(
        self, file_path: Union[Path, str], extension: str = ".npy"
    ) -> paddle.Tensor:
        if isinstance(file_path, str):
            file_path = self.index_to_file(file_path, extension)
        assert file_path.exists(), f"File path {file_path} does not exist"
        if extension == ".npy":
            float_max = np.finfo(np.float32).max
            float_min = np.finfo(np.float32).min
            data_double = np.load(str(file_path))
            data_clipped = np.clip(data_double, float_min, float_max * 1e-20)
            assert not np.isinf(data_clipped).any(), "存在溢出值！"
            assert not np.isnan(data_clipped).any(), "存在无效值！"
            data = paddle.to_tensor(data_clipped.astype(np.float32))

        elif extension == ".pdparams":
            data = paddle.load(path=str(str(file_path)))
        return data


class PathDictDataset(paddle.io.Dataset, LoadFile):

    def __init__(
        self,
        path: str = None,
        query_points=None,
        indices: Optional[List[str]] = None,
        norms_dict: Optional[Dict[str, Callable]] = {},
        lazy_loading=True,
    ):
        LoadFile.__init__(self, path)
        assert path is not None, "path is None"
        self.path = Path(path)
        self.query_points = query_points
        self.indices = indices
        self.norms_dict = norms_dict
        self.lazy_loading = lazy_loading
        if not self.lazy_loading:
            self.all_return_dict = [self.get_item(i) for i in range(len(self.indices))]

    def get_item(self, index):
        t1 = default_timer()
        file_index = self.indices[index] if self.indices else str(index).zfill(4)

        return_dict = {}
        extension = ".pdparams"
        file_key = "info"
        return_dict["info"] = self.load_file(f"{file_key}_{file_index}", extension)
        return_dict["df"] = self.load_file(f"df_{file_index}")
        return_dict["df_query_points"] = paddle.to_tensor(data=self.query_points)
        return_dict["vertices"] = None
        reference_area = return_dict["info"]["reference_area"]
        areas = self.load_file(f"area_{file_index}")
        centroids = self.load_file(f"centroid_{file_index}")
        triangle_normals = self.load_file(f"normal_{file_index}")

        flow_directions = paddle.zeros_like(x=triangle_normals)
        flow_directions[:, 0] = -1
        mass_density = return_dict["info"]["density"]
        flow_speed = return_dict["info"]["velocity"]
        const = 2.0 / (mass_density * flow_speed**2 * reference_area)
        projection = paddle.sum(
            x=triangle_normals * flow_directions, axis=1, keepdim=False
        )
        return_dict["dragWeight"] = const * projection * areas
        return_dict["dragWeightWss"] = (const * flow_directions * areas[:, None]).T
        return_dict["areas"] = areas
        return_dict["centroids"] = centroids
        for key in self.norms_dict:
            if key in return_dict:
                return_dict[key] = self.norms_dict[key](return_dict[key])
        if "location" in self.norms_dict:
            if return_dict["vertices"] is not None:
                return_dict["vertices"] = self.norms_dict["location"](vertices)
            return_dict["centroids"] = self.norms_dict["location"](
                return_dict["centroids"]
            )
            return_dict["df_query_points"] = self.norms_dict["location"](
                return_dict["df_query_points"]
            ).transpose(perm=[3, 0, 1, 2])
        t2 = default_timer()
        return_dict["Data_loading_time"] = t2 - t1
        return return_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.lazy_loading:
            return self.get_item(index)
        else:
            return self.all_return_dict[index]


class BaseCFDDataModule(BaseDataModule):

    def __init__(self):
        super().__init__()

    @property
    def inference_data(self):
        return self._inference_data

    def encode(self, norm_fn, data: paddle.Tensor) -> paddle.Tensor:
        norm_fn.to(data.place)
        return norm_fn.encode(data)

    def decode(self, norm_fn, data: paddle.Tensor) -> paddle.Tensor:
        norm_fn.to(data.place)
        return norm_fn.decode(data)

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-06
    ) -> Tuple[List[float], List[float]]:
        # with open(data_dir / filename, 'r') as fp:
        with open(data_dir / "../../" / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds

    def location_normalization(
        self,
        locations: paddle.Tensor,
        min_bounds: Union[paddle.Tensor, List[float]],
        max_bounds: Union[paddle.Tensor, List[float]],
    ) -> paddle.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        if not isinstance(min_bounds, paddle.Tensor):
            min_bounds = paddle.to_tensor(data=min_bounds)
        if not isinstance(max_bounds, paddle.Tensor):
            max_bounds = paddle.to_tensor(data=max_bounds)
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations

    def info_normalization(
        self, info: dict, min_bounds: List[float], max_bounds: List[float]
    ) -> dict:
        """
        Normalize info to [0, 1].
        """
        for i, (k, v) in enumerate(info.items()):
            info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
        return info

    def area_normalization(
        self, area: paddle.Tensor, min_bounds: float, max_bounds: float
    ) -> paddle.Tensor:
        """
        Normalize area to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)


class SAEInferenceDataModule(BaseCFDDataModule):

    def __init__(
        self,
        data_dir,
        out_keys: List[str] = ["pressure"],
        out_channels: List[int] = [1],
        n_inference: int = 1,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        eps=0.01,
        lazy_loading=True,
        bounds_dir=None,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir
        self.out_keys = out_keys
        self.out_channels = out_channels
        self.query_points = query_points
        self.spatial_resolution = spatial_resolution
        self.eps = eps
        self.lazy_loading = lazy_loading
        self.get_indices(n_inference)
        self.get_norms(bounds_dir)
        self.get_data()

    def load_ids(self, idx_path: str):
        idx_str_lst = []
        with open(idx_path, "r") as file:
            line = file.readline()
            while line:
                line = line.strip()
                idx_str_lst.append(line.split("_")[-1])
                line = file.readline()
        return idx_str_lst

    def init_idx(self, n_data, mode, filename) -> List[str]:
        idx_path = self.data_dir / filename
        if idx_path.exists():
            indices = self.load_ids(idx_path)
            paddle.sort(x=indices), paddle.argsort(x=indices)
            assert n_data <= len(
                indices
            ), f"only {len(indices)} meshes are available, but {n_data} are requested."
            indices = indices[:n_data]
        else:
            all_files = os.listdir(self.data_dir)
            prefix = "area"
            indices = [item[5:9] for item in all_files if item.startswith(prefix)]

            def extract_number(s):
                return int(s)

            indices.sort(key=extract_number)
            indices = indices[:n_data]

            full_caseids = [os.path.basename(self.data_dir)]
            # full_caseids = full_caseids[:n_data]
        return indices, full_caseids

    def get_indices(self, n_inference):
        self.inference_indices, self.inference_full_caseids = self.init_idx(
            n_inference, "inference", "inference_design_ids.txt"
        )

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-06
    ) -> Tuple[List[float], List[float]]:
        # with open(data_dir / filename, 'r') as fp:
        with open(os.path.join(data_dir, filename), "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds

    def get_norms(self, data_dir):
        min_bounds, max_bounds = self.load_bound(
            data_dir, filename="global_bounds.txt", eps=self.eps
        )
        min_info_bounds, max_info_bounds = self.load_bound(
            data_dir, filename="info_bounds.txt", eps=0.0
        )
        min_area_bound, max_area_bound = self.load_bound(
            data_dir, filename="area_bounds.txt", eps=0.0
        )
        if self.query_points is None:
            assert (
                self.spatial_resolution is not None
            ), "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], self.spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], self.spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], self.spatial_resolution[2])
            self.query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        location_norm_fn = lambda x: self.location_normalization(
            x, min_bounds, max_bounds
        )
        info_norm_fn = lambda x: self.info_normalization(
            x, min_info_bounds, max_info_bounds
        )
        area_norm_fn = lambda x: self.area_normalization(
            x, min_area_bound[0], max_area_bound[0]
        )
        self.norms_dict = {"location": location_norm_fn, "area": area_norm_fn}
        self.output_normalization = []
        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            if key == "pressure":
                data = np.zeros((100,))
            elif key == "wallshearstress":
                data = np.zeros((100, 3))

            key_normalization = UnitGaussianNormalizer(
                paddle.to_tensor(data=data), eps=1e-06, reduce_dim=[0], verbose=False
            )

            mean_std_filename = f"train_{key}_mean_std.txt"
            mean, std = self.load_bound(data_dir, filename=mean_std_filename, eps=0.0)
            key_normalization.mean, key_normalization.std = paddle.to_tensor(
                data=mean[: self.out_channels[i]]
            ), paddle.to_tensor(data=std[: self.out_channels[i]])
            self.norms_dict[key] = copy.deepcopy(key_normalization).encode
            self.output_normalization.append(key_normalization)

    def init_data(self, indices, mode="inference"):
        data_keys = ["info"]
        data_dict = PathDictDataset(
            path=self.data_dir,
            query_points=self.query_points,
            indices=indices,
            norms_dict=self.norms_dict,
            lazy_loading=self.lazy_loading,
        )
        return data_dict

    def get_data(self):
        self._inference_data = self.init_data(self.inference_indices, "inference")
        self._aggregatable = ["df", "df_query_points"]

    def load_file(self, file_path: Path) -> np.ndarray:
        assert file_path.exists(), f"File {file_path} does not exist"
        data = np.load(file_path).astype(np.float32)
        return data

    def decode(self, data, idx: int) -> paddle.Tensor:
        return super().decode(self.output_normalization[idx], data.T).T

    def collate_fn(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: paddle.stack(x=[data_dict[key] for data_dict in batch])}
            )
        remaining = list(set(batch[0].keys()) - set(self._aggregatable))
        for key in remaining:
            aggr_dict.update({key: [data_dict[key] for data_dict in batch]})
        return aggr_dict


class TestData(unittest.TestCase):

    def __init__(self, methodName: str, data_path: str) -> None:
        super().__init__(methodName)
        self.data_path = data_path

    def test_ahmed(self):
        dm = SAEDataModule(
            self.data_path, n_train=10, n_test=10, spatial_resolution=(64, 64, 64)
        )
        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                if isinstance(v, paddle.Tensor):
                    print(k, tuple(v.shape))
                else:
                    print(k)
                    for j in range(len(v)):
                        if isinstance(v[j], dict):
                            print(v[j])
                        else:
                            print(tuple(v[j].shape))
            break


if __name__ == "__main__":
    data_dir_test = Path("~/datasets/geono/ahmed").expanduser()
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestData("test_ahmed", data_dir_test))
    unittest.TextTestRunner().run(test_suite)
