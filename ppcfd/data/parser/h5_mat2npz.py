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


from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import scipy.io as sio

from ppcfd.data.parser.base_parser import BaseTransition


class H5MatTransition(BaseTransition):
    """Transition class for .mat and .h5 file.

    Args:
        file_path (Union[str, Path]): path of .mat and .h5 file.
        save_path (Optional[str], optional): path of saved .npz file. Defaults to None.
        save_data (Optional[bool], optional): Whether to save the .npz file. Defaults to False.
    """

    file_format = ["mat", "h5"]
    # be empty because `file_path` could be set by loader and all other prarmeters have default value
    required_params = []

    def __init__(
        self,
        file_path: Union[str, Path],
        save_path: Optional[str] = None,
        save_data: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(file_path)
        if save_path is not None:
            self.output_path = Path(save_path).absolute()
        else:
            self.output_path = self.file_path

        self.format = self.file_path.suffix[1:].lower()

        try:
            self.data = self.load_data()
        except Exception as e:
            raise ValueError(f"Failed to parse the file: {e}")

        if save_data:
            self.save_npz()

    def load_data(self):
        if self.format == "h5":
            return self.load_h5()
        elif self.format == "mat":
            try:
                return self.load_h5()
            except Exception:
                print("The .mat created by MATLAB versions < 7.3.")
                return self.load_mat()

    def load_mat(self):
        # MATLAB < v7.3
        raw_data = sio.loadmat(self.file_path)
        raw_data.pop("__header__", None)
        raw_data.pop("__version__", None)
        raw_data.pop("__globals__", None)
        return {k: self._parse_mat_struct(v) for k, v in raw_data.items()}

    def load_h5(self):
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
        output_path = self.output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.data)

    def get_data(self):
        return self.data
