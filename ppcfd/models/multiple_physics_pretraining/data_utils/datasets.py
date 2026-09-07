import paddle

"""
Remember to parameterize the file paths eventually
"""
import numpy as np

from .hdf5_datasets import *
from .mixed_dset_sampler import MultisetSampler

import glob

broken_paths = []
DSET_NAME_TO_OBJECT = {
    "swe": SWEDataset,
    "incompNS": IncompNSDataset,
    "diffre2d": DiffRe2DDataset,
    "compNS": CompNSDataset,
}


def get_data_loader(params, paths, distributed, split="train", rank=0, train_offset=0):
    dataset = MixedDataset(
        paths,
        n_steps=params.n_steps,
        train_val_test=params.train_val_test,
        split=split,
        tie_fields=params.tie_fields,
        use_all_fields=params.use_all_fields,
        enforce_max_steps=params.enforce_max_steps,
        train_offset=train_offset,
    )
    sampler = MultisetSampler(
        dataset,
        params.batch_size,
        shuffle=(split == "train"),
        distributed=distributed,
        max_samples=params.epoch_size,
        rank=rank,
        world_size=paddle.distributed.get_world_size() if distributed else 1,
    )
    batch_sampler = paddle.io.BatchSampler(
        sampler=sampler,
        batch_size=int(params.batch_size),
        drop_last=True,
    )
    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=params.num_data_workers,
        return_list=True,
    )
    return dataloader, dataset, sampler


class MixedDataset(paddle.io.Dataset):
    def __init__(
        self,
        path_list=[],
        n_steps=1,
        dt=1,
        train_val_test=(0.8, 0.1, 0.1),
        split="train",
        tie_fields=True,
        use_all_fields=True,
        extended_names=False,
        enforce_max_steps=False,
        train_offset=0,
    ):
        super().__init__()
        self.train_offset = train_offset
        self.path_list, self.type_list, self.include_string = zip(*path_list)
        self.tie_fields = tie_fields
        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test
        self.use_all_fields = use_all_fields
        for dset, path, include_string in zip(
            self.type_list, self.path_list, self.include_string
        ):
            subdset = DSET_NAME_TO_OBJECT[dset](
                path,
                include_string,
                n_steps=n_steps,
                dt=dt,
                train_val_test=train_val_test,
                split=split,
            )
            try:
                len(subdset)
            except ValueError:
                raise ValueError(
                    f"Dataset {path} is empty. Check that n_steps < trajectory_length in file."
                )
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1] + len(self.sub_dsets[-1]))
        self.offsets[0] = -1
        self.subset_dict = self._build_subset_dict()

    def get_state_names(self):
        name_list = []
        if self.use_all_fields:
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                name_list += field_names
            return name_list
        else:
            visited = set()
            for dset in self.sub_dsets:
                name = dset.get_name()
                if not name in visited:
                    visited.add(name)
                    name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl]

    def _build_subset_dict(self):
        if self.tie_fields:
            subset_dict = {
                "swe": [3],
                "incompNS": [0, 1, 2],
                "compNS": [0, 1, 2, 3],
                "diffre2d": [4, 5],
            }
        elif self.use_all_fields:
            cur_max = 0
            subset_dict = {}
            for name, dset in DSET_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                subset_dict[name] = list(range(cur_max, cur_max + len(field_names)))
                cur_max += len(field_names)
        else:
            subset_dict = {}
            cur_max = self.train_offset
            for dset in self.sub_dsets:
                name = dset.get_name(self.extended_names)
                if not name in subset_dict:
                    subset_dict[name] = list(
                        range(cur_max, cur_max + len(dset.field_names))
                    )
                    cur_max += len(dset.field_names)
        return subset_dict

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.offsets, index, side="right") - 1
        local_idx = index - max(self.offsets[file_idx], 0)
        try:
            x, bcs, y = self.sub_dsets[file_idx][local_idx]
        except Exception as err:
            current_rank = (
                int(paddle.distributed.get_rank())
                if paddle.distributed.is_initialized()
                else 0
            )
            raise RuntimeError(
                f"FAILED AT file_idx={file_idx} local_idx={local_idx} index={index} rank={current_rank}"
            ) from err
        return (
            x,
            file_idx,
            paddle.to_tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]),
            bcs,
            y,
        )

    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])
