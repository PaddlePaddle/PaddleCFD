# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Authors: lijialin03(lijialin03@baidu.com)
Date:    2025/04/01
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import scipy.io as sio


class MatTransition:
    """Transition class for .mat file.

    Args:
        file_path (str): path of .mat file.
        save_path (Optional[str], optional): path of saved .npz file. Defaults to None.
        save_data (Optional[bool], optional): Whether to save the .npz file. Defaults to True.
    """

    def __init__(
        self,
        file_path: str,
        save_path: Optional[str] = None,
        save_data: Optional[bool] = True,
    ):
        super().__init__()
        self.file_path = file_path
        self.save_path = save_path

        try:
            self.data = self.load_data()
        except Exception as e:
            raise ValueError(f"Failed to parse the file: {e}")

        if save_data:
            self.save_npz()

    def load_data(self):
        try:
            # MATLAB < v7.3
            raw_data = sio.loadmat(self.file_path)
            raw_data.pop("__header__", None)
            raw_data.pop("__version__", None)
            raw_data.pop("__globals__", None)
            return {k: self._parse_mat_struct(v) for k, v in raw_data.items()}
        except NotImplementedError:
            print("The .mat created by MATLAB versions >= 7.3, try to load it using h5py instead.")
            with h5py.File(self.file_path, "r") as f:
                return {key: self._parse_hdf5_group(f[key]) for key in f.keys()}

    def _parse_mat_struct(self, struct):
        if isinstance(struct, np.ndarray):
            if struct.dtype == np.dtype("object"):
                if struct.size == 1:
                    item = struct.item()
                    processed = self._parse_mat_struct(item)
                    if isinstance(item, np.ndarray) and item.dtype.names:
                        return {name: self._parse_mat_struct(item[name]) for name in item.dtype.names}
                    else:
                        return processed
                else:
                    return [self._parse_mat_struct(item) for item in struct]
            elif struct.dtype.names:
                if struct.size == 1:
                    return {name: self._parse_mat_struct(struct[name]) for name in struct.dtype.names}
                else:
                    return [
                        {name: self._parse_mat_struct(struct[name][i]) for name in struct.dtype.names}
                        for i in range(struct.size)
                    ]
        return struct

    def _parse_hdf5_group(self, group):
        if isinstance(group, h5py.Dataset):
            value = group[()]
            if isinstance(value, np.ndarray) and value.dtype == np.uint16:
                try:
                    return bytes(value).decode("utf-16").strip("\x00")
                except Exception:
                    pass
            if isinstance(value, h5py.Reference):
                ref_group = group.parent[value]
                return [self._parse_hdf5_group(ref_group[i]) for i in range(len(ref_group))]
            return value
        elif isinstance(group, h5py.Group):
            return {k: self._parse_hdf5_group(v) for k, v in group.items()}
        return group

    def save_npz(self):
        if self.save_path is not None:
            output_path = Path(self.save_path)
        else:
            output_path = Path(self.file_path)
        output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.data)


if __name__ == "__main__":
    path = "./burgers.mat"
    # save_path = './burgers.npz'
    # save_path = './burgers'
    # trans_obj = MatTransition(path, save_path)
    trans_obj = MatTransition(path)
    for key, value in trans_obj.data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape: {value.shape}")
        else:
            print(f"{key}: type: {type(value)}")
