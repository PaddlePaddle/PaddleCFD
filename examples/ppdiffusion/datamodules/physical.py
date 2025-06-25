import itertools
import logging
from collections import defaultdict
from typing import Dict

import numpy as np
import paddle.distributed as dist
from einops import rearrange
from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import Dataset
from paddle.io import DistributedBatchSampler
from tqdm import tqdm

import ppcfd


class PhysicalDatastet(Dataset):
    def __init__(
        self,
        data_dir: str,
        physical_system: str = "navier-stokes",
        horizon: int = 1,
        window: int = 1,
        multi_horizon: bool = False,
        num_trajectories: int = None,
        condition_name: str = "condition",  # this should be the same as in the model (the keyword argument)
        **kwargs,
    ):
        super(PhysicalDatastet, self).__init__()
        assert window == 1, "window > 1 is not supported yet for this data module."

        self.data_dir = data_dir
        self.horizon = horizon
        self.window = window

        self.multi_horizon = multi_horizon
        self.num_trajectories = num_trajectories
        self.physical_system = physical_system
        self.condition_name = condition_name

        self.dataset_len = 0
        self.load_data()

    def _check_args(self):
        h = self.horizon
        w = self.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w > 0, f"window must be > 0, but is {w}"

    def load_data(self):
        """Load data. Set internal variables: self.dataset, self._data_val, self._data_test."""
        dataset = ppcfd.data.load_dataset(self.data_dir + ".h5")[0]

        if self.multi_horizon:
            numpy_tensors = self.create_dataset_multi_horizon(dataset, False)
        else:
            numpy_tensors = self.create_dataset_single_horizon(dataset, False)

        self.dataset = numpy_tensors
        assert self.dataset is not None, "Could not create dataset"

    def create_dataset_single_horizon(self, dataset, keep_trajectory_dim: bool = False) -> Dict[str, np.ndarray]:
        """Create a dataset from the given dataset and return it."""
        data = self.create_dataset_multi_horizon(dataset, keep_trajectory_dim)
        dynamics = np.concatenate(data["dynamics"], axis=0).astype(np.float32)
        window, horizon = self.window, self.horizon
        assert (
            tuple(dynamics.shape)[1] == window + horizon
        ), f"Expected dynamics to have shape (b, {window + horizon}, ...)"
        inputs = dynamics[:, :window, ...]
        targets = dynamics[:, -1, ...]
        return {"inputs": inputs, "targets": targets, **data}

    def create_dataset_multi_horizon(self, dataset, keep_trajectory_dim: bool = False) -> Dict[str, np.ndarray]:
        """Create a numpy dataset from the given xarray dataset and return it."""
        # dataset is 4D tensor with dimensions (grid-box, time, lat, lon)
        # Create a tensor, X, of shape (batch-dim, horizon, lat, lon),
        # where each X[i] is a temporal sequence of horizon time steps
        window, horizon = self.window, self.horizon
        trajectories = defaultdict(list)
        # go through all trajectories and concatenate them in the 2nd dimension (new axis)
        n_trajectories = len(dataset) if self.num_trajectories is None else min(len(dataset), self.num_trajectories)

        logging.info(f"Loading data from {self.data_dir}")
        for k, traj_i in tqdm(itertools.islice(dataset.items(), n_trajectories), total=n_trajectories, desc="Loading"):
            traj_len = traj_i["trajectory_meta"]["num_time_steps"]
            time_len = traj_len - horizon - window + 1

            dynamics_i = traj_i["features"]
            assert (
                dynamics_i.shape[0] == traj_len
            ), f"Error: shape {dynamics_i.shape} not equal to {traj_len} along axis 0"

            # Repeat extra_fixed_mask for each example in the trajectory (it is the same for all examples)
            extra_fixed_mask = np.repeat(np.expand_dims(traj_i["condition"], axis=0), time_len, axis=0)

            # To save memory, we create the dataset through sliding window views
            dynamics_i = np.lib.stride_tricks.sliding_window_view(dynamics_i, time_len, axis=0)
            dynamics_i = rearrange(dynamics_i, "horizon c h w example -> example horizon c h w")
            assert (
                dynamics_i.shape[0] == time_len
            ), f"Error: shape {dynamics_i.shape} not equal to {time_len} along axis 0"
            assert (
                extra_fixed_mask.shape[0] == time_len
            ), f"Error: shape {extra_fixed_mask.shape} not equal to {time_len} along axis 0"
            if keep_trajectory_dim:
                dynamics_i = np.expand_dims(dynamics_i, axis=0)
                extra_fixed_mask = np.expand_dims(extra_fixed_mask, axis=0)
            # add to the dataset
            traj_i["trajectory_meta"]["t"] = traj_i["t"]
            traj_i["trajectory_meta"]["fixed_mask"] = traj_i["fixed_mask"]
            if self.physical_system == "navier-stokes":
                traj_i["trajectory_meta"]["vertices"] = traj_i["vertices"]

            traj_metadata = [traj_i["trajectory_meta"]] * time_len
            trajectories["dynamics"].append(dynamics_i.astype(np.float32))
            trajectories[self.condition_name].append(extra_fixed_mask.astype(np.float32))
            trajectories["metadata"].extend(traj_metadata)

            self.dataset_len += dynamics_i.shape[0]

        # do not concatenate here to avoid excessive memory usage
        for key in ["dynamics", self.condition_name]:
            if trajectories[key]:
                trajectories[key] = np.concatenate(trajectories[key], axis=0).astype(np.float32)

        # print(f'Shapes={trajectories["dynamics"].shape}, {trajectories["extra_condition"].shape}')
        # E.g. with 90 total examples, horizon=5, window=1: Shapes=(90, 6, 3, 221, 42), (90, 2, 221, 42)
        return trajectories

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return {key: self.dataset[key][idx] for key in self.dataset.keys()}


class PhysicalDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def dataloader(self, batch_size, shuffle=False, drop_last=False, num_workers=0, **kwargs):
        if not batch_size:
            batch_size = len(self.dataset)

        try:
            world_size = dist.get_world_size()
            use_distributed = world_size > 1
        except Exception:
            use_distributed = False
        sampler_cls = DistributedBatchSampler if use_distributed else BatchSampler

        batch_sampler = sampler_cls(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            use_shared_memory=True,
            **kwargs,
        )
