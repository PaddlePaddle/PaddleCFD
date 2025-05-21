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


from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import wandb


Number = Union[int, float]


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-05, reduce_dim=0, verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim
        y = x.numpy()
        self.mean = paddle.to_tensor(np.mean(y, axis=reduce_dim, keepdims=True).squeeze(axis=0))
        self.std = paddle.to_tensor(np.std(y, axis=reduce_dim, keepdims=True).squeeze(axis=0))

        self.eps = eps
        if verbose:
            print(f"UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.")
            print(f"   Mean and std of shape {self.mean.shape}, eps={eps}")

    def encode(self, x):
        x -= self.mean
        x /= self.std + self.eps
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        x *= std
        x += mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def count_params(model):
    """Returns the number of parameters of a paddle model"""
    return sum([(p.size * 2 if p.is_complex() else p.size) for p in model.parameters()])


def wandb_login(api_key_file="../config/wandb_api_key.txt", key=None):
    if key is None:
        key = get_wandb_api_key(api_key_file)
    wandb.login(key=key)


def set_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        return [[float(s)] * n_dim for s in scaling_factor]
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None
