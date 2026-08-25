import paddle

"""
Process CFDBench data. 
Source: https://github.com/HaoZhongkai/DPOT
"""
from pathlib import Path

import h5py
import numpy as np
try:
    from examples.prose_fd.data_utils.cfdbench import get_auto_dataset
except ImportError:
    from cfdbench import get_auto_dataset


def preprocess_cfdbench_data():
    train_data_cavity, dev_data_cavity, test_data_cavity = get_auto_dataset(
        data_dir=Path("/data/shared/dataset/cfdbench"),
        data_name="cavity_prop_bc_geo",
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    train_data_cylinder, dev_data_cylinder, test_data_cylinder = get_auto_dataset(
        data_dir=Path("/data/shared/dataset/cfdbench"),
        data_name="cylinder_prop_bc_geo",
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    train_data_tube, dev_data_tube, test_data_tube = get_auto_dataset(
        data_dir=Path("/data/shared/dataset/cfdbench"),
        data_name="tube_prop_bc_geo",
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
    )
    cavity_lens = [data.shape[0] for data in train_data_cavity.all_features]
    cylinder_lens = [data.shape[0] for data in train_data_cylinder.all_features]
    tube_lens = [data.shape[0] for data in train_data_tube.all_features]
    train_cavity_feats, train_cylinder_feats, train_tube_feats = (
        train_data_cavity.all_features,
        train_data_cylinder.all_features,
        train_data_tube.all_features,
    )
    test_cavity_feats, test_cylinder_feats, test_tube_feats = (
        test_data_cavity.all_features,
        test_data_cylinder.all_features,
        test_data_tube.all_features,
    )
    dev_cavity_feats, dev_cylinder_feats, dev_tube_feats = (
        dev_data_cavity.all_features,
        dev_data_cylinder.all_features,
        dev_data_tube.all_features,
    )
    train_feats = train_cavity_feats + train_cylinder_feats + train_tube_feats
    test_feats = test_cavity_feats + test_cylinder_feats + test_tube_feats
    dev_feats = dev_cavity_feats + dev_cylinder_feats + dev_tube_feats
    print()
    print(cavity_lens)
    print(cylinder_lens)
    print(tube_lens)
    print()
    infer_steps = 20

    def split_trajectory(data_list, time_step, grid_size=64):
        traj_split = []
        for i, x in enumerate(data_list):
            T = x.shape[0]
            num_segments = int(np.floor(T / time_step))
            if num_segments < 1:
                continue
            padded_length = num_segments * time_step
            padded_array = x[:padded_length]
            padded_array = paddle.nn.functional.interpolate(
                paddle.from_numpy(padded_array).float(),
                size=(grid_size, grid_size),
                mode="bilinear",
                align_corners=True,
            ).numpy()
            padded_array = padded_array.reshape(
                (num_segments, time_step, *padded_array.shape[1:])
            )
            traj_split.append(padded_array)
        traj_split = np.concatenate(traj_split, axis=0)
        return traj_split

    train_data = split_trajectory(train_feats, infer_steps, grid_size=128)
    test_data = split_trajectory(test_feats, infer_steps, grid_size=128)
    dev_data = split_trajectory(dev_feats, infer_steps, grid_size=128)
    train_data, test_data, dev_data = (
        train_data.transpose(0, 1, 3, 4, 2),
        test_data.transpose(0, 1, 3, 4, 2),
        dev_data.transpose(0, 1, 3, 4, 2),
    )
    print(train_data.shape, test_data.shape, dev_data.shape)
    with h5py.File("/data/shared/dataset/cfdbench/ns2d_cdb_train.hdf5", "w") as fp:
        fp.create_dataset("data", data=train_data, compression=None)
    with h5py.File("/data/shared/dataset/cfdbench/ns2d_cdb_test.hdf5", "w") as fp:
        fp.create_dataset("data", data=test_data, compression=None)
    with h5py.File("/data/shared/dataset/cfdbench/ns2d_cdb_val.hdf5", "w") as fp:
        fp.create_dataset("data", data=dev_data, compression=None)


if __name__ == "__main__":
    preprocess_cfdbench_data()
