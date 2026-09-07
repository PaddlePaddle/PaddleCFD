import os

import einops
import numpy as np
import paddle
import scipy
from distributed.config import barrier, is_data_rank0
from kappadata.copying.image_folder import \
    copy_imagefolder_from_global_to_local
from kappautils.param_checking import to_2tuple
from paddle_utils import *
from paddle_geometric.nn.pool import radius, radius_graph
from utils.num_worker_heuristic import get_fair_cpu_count

from .base.dataset_base import DatasetBase


class CfdDataset(DatasetBase):
    def __init__(
        self,
        version,
        num_input_timesteps,
        radius_graph_r=None,
        radius_graph_max_num_neighbors=None,
        num_input_points=None,
        num_input_points_ratio=None,
        num_input_points_mode="uniform",
        num_supernodes=None,
        supernode_edge_mode="mesh_to_supernode",
        num_query_points=None,
        num_query_points_mode="input",
        couple_query_with_input=False,
        split="train",
        standardize_query_pos=False,
        global_root=None,
        local_root=None,
        grid_resolution=None,
        max_num_sequences=None,
        max_num_timesteps=None,
        norm="mean0std1",
        clamp=None,
        clamp_mode="hard",
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.version = version
        self.split = split
        self.max_num_sequences = max_num_sequences
        self.max_num_timesteps = max_num_timesteps
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors or int(
            10000000000.0
        )
        if self.radius_graph_max_num_neighbors == float("inf"):
            self.radius_graph_max_num_neighbors = int(10000000000.0)
        self.num_query_points = num_query_points
        self.num_query_points_mode = num_query_points_mode
        self.couple_query_with_input = couple_query_with_input
        if couple_query_with_input:
            assert (
                self.num_query_points is None
            ), "couple_query_inputs requires 'num_query_points is None'"
        self.num_input_points = to_2tuple(num_input_points)
        self.num_input_points_ratio = to_2tuple(num_input_points_ratio)
        self.num_input_points_mode = num_input_points_mode
        self.num_supernodes = num_supernodes
        self.supernode_edge_mode = supernode_edge_mode
        assert not (
            self.num_input_points is not None
            and self.num_input_points_ratio is not None
        )
        assert grid_resolution is None or len(grid_resolution) == 2
        self.grid_resolution = grid_resolution
        assert 0 < num_input_timesteps
        self.num_input_timesteps = num_input_timesteps
        self.standardize_query_pos = standardize_query_pos
        self.seed = seed
        self.norm = norm
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        self.num_input_points_cache = []
        if norm == "none":
            self.mean = paddle.tensor([0.0, 0.0, 0.0])
            self.std = paddle.tensor([1.0, 1.0, 1.0])
        elif version == "v1-1sim":
            self.mean = paddle.tensor(
                [0.029124850407242775, 0.00255209649913013, 0.0010001148330047727]
            )
            self.std = paddle.tensor(
                [0.026886435225605965, 0.01963668502867222, 0.001666962169110775]
            )
        elif version == "v1-2sims":
            self.mean = paddle.tensor(
                [0.029124850407242775, 0.00255209649913013, 0.0010001148330047727]
            )
            self.std = paddle.tensor(
                [0.026886435225605965, 0.01963668502867222, 0.001666962169110775]
            )
        elif version == "v1-1000sims":
            self.mean = paddle.tensor(
                [0.036486752331256866, 2.509498517611064e-05, 0.000451924919616431]
            )
            self.std = paddle.tensor(
                [0.026924047619104385, 0.02058381214737892, 0.002078353427350521]
            )
        elif version == "v1-10000sims":
            self.mean = paddle.tensor(
                [0.0152587890625, -1.7881393432617188e-06, 0.0003612041473388672]
            )
            self.std = paddle.tensor(
                [0.0233612060546875, 0.0184173583984375, 0.0019378662109375]
            )
        elif version == "v1-686sims-1object":
            self.mean = paddle.tensor(
                [0.03460693359375, -3.236532211303711e-05, 7.647275924682617e-05]
            )
            self.std = paddle.tensor(
                [0.01055145263671875, 0.00829315185546875, 0.0004229545593261719]
            )
        elif version == "v1-1900sims":
            self.mean = paddle.tensor(
                [0.03460693359375, -1.806020736694336e-05, 0.00010699033737182617]
            )
            self.std = paddle.tensor(
                [0.01363372802734375, 0.01102447509765625, 0.0006461143493652344]
            )
        elif version == "v2-2500sims":
            if norm == "mean0std1q25":
                self.mean = paddle.tensor(
                    [
                        0.03450389206409454,
                        -5.949020305706654e-06,
                        0.00010136327182408422,
                    ]
                )
                self.std = paddle.tensor(
                    [
                        0.0031622101087123156,
                        0.0018765029963105917,
                        0.0001263884623767808,
                    ]
                )
            else:
                raise NotImplementedError
        elif version == "v2-5000sims":
            if norm == "mean0std1":
                self.mean = paddle.tensor(
                    [0.0258941650390625, -1.823902130126953e-05, 0.00012934207916259766]
                )
                self.std = paddle.tensor(
                    [0.01482391357421875, 0.01200103759765625, 0.0007719993591308594]
                )
            elif norm == "mean0std1q05":
                self.mean = paddle.tensor(
                    [
                        0.035219039767980576,
                        -2.1968364308122545e-05,
                        0.0001966722047654912,
                    ]
                )
                self.std = paddle.tensor(
                    [0.010309861041605473, 0.007318499963730574, 0.0005381687660701573]
                )
            elif norm == "mean0std1q1":
                self.mean = paddle.tensor(
                    [
                        0.036569397896528244,
                        -2.364995816606097e-05,
                        0.00019191036699339747,
                    ]
                )
                self.std = paddle.tensor(
                    [0.00839781854301691, 0.005956545472145081, 0.0004608448361977935]
                )
            elif norm == "mean0std1q25":
                self.mean = paddle.tensor(
                    [
                        0.036188144236803055,
                        -2.3106376829673536e-05,
                        0.0001511715818196535,
                    ]
                )
                self.std = paddle.tensor(
                    [
                        0.0047589014284312725,
                        0.0034182844683527946,
                        0.00027269049314782023,
                    ]
                )
            else:
                raise NotImplementedError
        elif version == "v2-6000sims":
            if norm == "mean0std1q25":
                self.mean = paddle.tensor(
                    [
                        0.026319274678826332,
                        -1.2412071725975693e-07,
                        5.59896943741478e-05,
                    ]
                )
                self.std = paddle.tensor(
                    [0.0031868498772382736, 0.0021304511465132236, 0.000102771315141581]
                )
            else:
                raise NotImplementedError
        elif version == "v3-10000sims":
            if norm == "mean0std1q25":
                self.mean = paddle.tensor(
                    [0.03648518770933151, 1.927249059008318e-06, 0.000112384237581864]
                )
                self.std = paddle.tensor(
                    [0.005249467678368092, 0.003499444341287017, 0.0002817418717313558]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.max_x_pos = 200
        self.max_y_pos = 300
        self.pos_scale = 200
        self.sim_x_pos_min = -0.5
        self.sim_y_pos_min = -0.5
        global_root, local_root = self._get_roots(
            global_root, local_root, "mesh_dataset"
        )
        if local_root is None:
            self.source_root = global_root / version
            self.logger.info(f"data_source (global): '{self.source_root}'")
        else:
            self.source_root = local_root / "mesh_dataset"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{self.source_root}'")
                copy_imagefolder_from_global_to_local(
                    global_path=global_root,
                    local_path=self.source_root,
                    relative_path=version,
                    log_fn=self.logger.info,
                    num_workers=min(10, get_fair_cpu_count()),
                )
            self.source_root = self.source_root / version
            barrier()
        assert (
            self.source_root.exists()
        ), f"'{self.source_root.as_posix()}' doesn't exist"
        seqnames = list(
            sorted(
                [
                    name
                    for name in os.listdir(self.source_root)
                    if (self.source_root / name).is_dir()
                ]
            )
        )
        assert (
            len(seqnames) > 0
        ), f"couldnt find any sequences in '{self.source_root.as_posix()}'"
        seqnames = self._filter_split_seqnames(seqnames)
        assert (
            len(seqnames) > 0
        ), f"filtered out all sequences of '{self.source_root.as_posix()}'"
        self.samples = []
        for seqname in seqnames:
            samples = [
                fname
                for fname in sorted(os.listdir(self.source_root / seqname))
                if self._is_timestep_fname(fname)
            ]
            self.samples.append((seqname, samples))
        seqlens = [len(fnames) for _, fnames in self.samples]
        if not all(seqlens[0] == seqlen for seqlen in seqlens):
            for seqname, fnames in self.samples:
                self.logger.info(f"- {seqname} {len(fnames)}")
            raise RuntimeError("not all sequencelengths are the same")
        if self.max_num_timesteps is not None:
            assert max_num_timesteps <= seqlens[0]
            self.max_timestep = max_num_timesteps
        else:
            self.max_timestep = seqlens[0]
        self.max_timestep -= 1

    def _filter_split_seqnames(self, seqnames):
        if self.version in ["v1-1sim"]:
            assert len(seqnames) == 1
            return seqnames
        if self.version in ["v1-2sims"]:
            assert len(seqnames) == 2
            return seqnames
        if self.version in ["v1-1000sims"]:
            assert self.max_num_sequences is None
            seqname_to_caseidx = {
                seqname: int(seqname.split("_")[1]) for seqname in seqnames
            }
            if self.split == "train":
                return [
                    seqname for seqname, idx in seqname_to_caseidx.items() if 10 <= idx
                ]
            if self.split == "test":
                return [
                    seqname for seqname, idx in seqname_to_caseidx.items() if idx < 10
                ]
            if self.split == "train-10sims":
                return [
                    seqname for seqname, idx in seqname_to_caseidx.items() if 10 <= idx
                ][:10]
        if self.version in ["v1-10000sims"]:
            assert self.max_num_sequences is None
            seqname_to_caseidx = {
                seqname: int(seqname.split("_")[1]) for seqname in seqnames
            }
            if self.split == "train":
                return [
                    seqname
                    for seqname, idx in seqname_to_caseidx.items()
                    if idx <= 10005
                ]
            if self.split == "test":
                return [
                    seqname
                    for seqname, idx in seqname_to_caseidx.items()
                    if idx > 10005
                ]
            if self.split == "train-10sims":
                return [
                    seqname for seqname, idx in seqname_to_caseidx.items() if idx < 10
                ]
        if self.version in ["v1-686sims-1object"]:
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            num_train_sequences = 650
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        if self.version in ["v1-1900sims"]:
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            num_train_sequences = 1900
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[num_train_sequences:]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-5000sims":
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) >= 5500
            num_train_sequences = 5000
            num_test_sequences = 500
            num_val_sequences = 500
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[
                        num_train_sequences : num_train_sequences + num_test_sequences
                    ]
                ]
            elif self.split == "val":
                start = num_train_sequences + num_test_sequences
                end = start + num_val_sequences
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[start:end]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-10000sims":
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) >= 10000
            num_train_sequences = 9500
            num_test_sequences = 500
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[
                        num_train_sequences : num_train_sequences + num_test_sequences
                    ]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-2500sims":
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) >= 2500
            num_train_sequences = 2000
            num_test_sequences = 500
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[
                        num_train_sequences : num_train_sequences + num_test_sequences
                    ]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        if self.version == "v2-6000sims":
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) == 6000
            num_train_sequences = 5000
            num_test_sequences = 1000
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[
                        num_train_sequences : num_train_sequences + num_test_sequences
                    ]
                ]
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        if self.version == "v3-10000sims":
            caseidx_to_seqname = {
                int(seqname.split("_")[1]): seqname for seqname in seqnames
            }
            sorted_caseidxs = list(sorted(caseidx_to_seqname.keys()))
            assert len(sorted_caseidxs) == 10000
            num_train_sequences = 8000
            num_valid_sequences = 1000
            num_test_sequences = 1000
            if self.split == "train":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[:num_train_sequences]
                ]
                assert len(split_seqnames) == num_train_sequences
            elif self.split == "valid":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[
                        num_train_sequences : num_train_sequences + num_valid_sequences
                    ]
                ]
                assert len(split_seqnames) == num_valid_sequences
            elif self.split == "test":
                split_seqnames = [
                    caseidx_to_seqname[case_idx]
                    for case_idx in sorted_caseidxs[
                        num_train_sequences + num_valid_sequences :
                    ]
                ]
                assert len(split_seqnames) == num_test_sequences
            else:
                raise NotImplementedError
            if self.max_num_sequences is not None:
                split_seqnames = split_seqnames[: self.max_num_sequences]
            return split_seqnames
        raise NotImplementedError

    @staticmethod
    def _is_timestep_fname(fname):
        if not fname.endswith(".th"):
            return False
        if fname in [
            "object_mask.th",
            "U_init.th",
            "x.th",
            "y.th",
            "movement_per_position.th",
            "num_objects.th",
        ]:
            return False
        if fname.startswith("edge_index"):
            return False
        if fname.startswith("sampling_weights"):
            return False
        assert fname.endswith("_mesh.th") and fname[: -len("_mesh.th")].isdigit()
        return True

    def __len__(self):
        if self.num_input_timesteps == float("inf"):
            return len(self.samples)
        return self.max_timestep * len(self.samples)

    def getitem_timestep(self, idx, ctx=None):
        return idx % self.max_timestep

    def getshape_timestep(self):
        return (self.max_timestep,)

    def denormalize(self, data, inplace=False):
        assert data.size(1) == len(self.mean)
        shape = [1] * data.ndim
        shape[1] = len(self.mean)
        mean = self.mean.view(*shape).to(data.device)
        std = self.std.view(*shape).to(data.device)
        if inplace:
            data.mul_(std).add_(mean)
        else:
            data = data * std + mean
        return data

    def _get_sim_name(self, idx):
        if self.num_input_timesteps == float("inf"):
            sim_name, _ = self.samples[idx]
        else:
            seqidx = idx // self.max_timestep
            sim_name, _ = self.samples[seqidx]
        return sim_name

    def getitem_geometry2d(self, idx, ctx=None):
        sim_name = self._get_sim_name(idx)
        return paddle.load(path=str(self.source_root / sim_name / "object_mask.th"))

    def getshape_geometry2d(self):
        shape = self.getitem_geometry2d(0).shape
        return 2, *shape

    def getitem_num_objects(self, idx, ctx=None):
        sim_name = self._get_sim_name(idx)
        return paddle.load(path=str(self.source_root / sim_name / f"num_objects.th"))

    def getitem_velocity(self, idx, ctx=None):
        if self.version in ["v1-1sim"]:
            return 0
        sim_name = self._get_sim_name(idx)
        if self.version in ["v1-2sims"]:
            if idx < len(self) // 2:
                return 0
            return 1
        v = paddle.load(path=str(self.source_root / sim_name / f"U_init.th"))
        v.sub_(0.01).div_(0.05).mul_(200)
        return v

    def _get_generator(self, idx):
        if self.num_input_timesteps == float("inf"):
            return paddle.Generator().manual_seed(int(idx) + (self.seed or 0))
        if self.split == "test":
            assert self.seed is not None
        if self.seed is not None:
            return paddle.Generator().manual_seed(int(idx) + self.seed)
        return None

    def _downsample_input(self, data, idx=None, ctx=None):
        if self.num_input_points_ratio is None and self.num_input_points is None:
            return data
        assert ctx is not None
        if "input_perm" in ctx:
            perm = ctx["input_perm"]
        else:
            generator = self._get_generator(idx)
            if self.num_input_points is not None:
                if self.num_input_points[0] == self.num_input_points[1]:
                    end = self.num_input_points[0]
                elif len(self.num_input_points_cache) > 0:
                    assert len(self.num_input_points_cache) == 1
                    end = self.num_input_points_cache.pop()
                else:
                    assert (
                        generator is None
                    ), "variable num_input_points doesnt support seed"
                    lb, ub = self.num_input_points
                    midpoint = paddle.randint(low=0, high=ub - lb, shape=(1,)).item()
                    end = ub - midpoint
                    self.num_input_points_cache.append(lb + midpoint)
            elif self.num_input_points_ratio is not None:
                if self.num_input_points_ratio[0] == self.num_input_points_ratio[1]:
                    end = int(len(data) * self.num_input_points_ratio[0])
                else:
                    lb, ub = self.num_input_points_ratio
                    num_points_ratio = paddle.rand(size=(1,)).item() * (ub - lb) + lb
                    end = int(len(data) * num_points_ratio)
            else:
                raise NotImplementedError
            if self.num_input_points_mode == "uniform":
                perm = paddle.randperm(len(data))[:end]
            else:
                sim_name = self._get_sim_name(idx)
                weights = paddle.load(
                    path=str(
                        self.source_root
                        / sim_name
                        / f"sampling_weights_{self.num_input_points_mode}.th"
                    )
                )
                perm = paddle.multinomial(
                    weights.float(), num_samples=end, replacement=False
                )
            ctx["input_perm"] = perm
        return data[perm]

    def _downsample_query(self, data, idx=None, ctx=None):
        if self.num_input_timesteps == float("inf"):
            assert self.num_query_points is None
            return self._downsample_input(data, idx=idx, ctx=ctx)
        if self.couple_query_with_input:
            assert self.num_query_points is None
            return self._downsample_input(data, idx=idx, ctx=ctx)
        if self.num_query_points is None:
            return data
        if "query_perm" in ctx:
            perm = ctx["query_perm"]
        else:
            if self.num_query_points_mode == "input":
                perm = ctx["input_perm"]
                assert len(perm) >= self.num_query_points
            elif self.num_query_points_mode == "arbitrary":
                generator = self._get_generator(idx)
                perm = paddle.randperm(len(data))
            else:
                raise NotImplementedError
            perm = perm[: self.num_query_points]
            ctx["query_perm"] = perm
        return data[perm]

    def _downsample_reconstruction_output(self, data, idx=None, ctx=None):
        assert not self.num_input_timesteps == float("inf")
        assert not self.couple_query_with_input
        if self.num_query_points is None:
            return data
        if "rec_perm" in ctx:
            perm = ctx["rec_perm"]
        else:
            if self.num_query_points_mode == "input":
                raise NotImplementedError
            elif self.num_query_points_mode == "arbitrary":
                generator = self._get_generator(idx)
                perm = paddle.randperm(len(data))
            else:
                raise NotImplementedError
            perm = perm[: self.num_query_points]
            ctx["rec_perm"] = perm
        return data[perm]

    def _load_xy(self, case_uri):
        x = paddle.load(path=str(case_uri / f"y.th")).float()
        y = paddle.load(path=str(case_uri / f"x.th")).float()
        x.sub_(self.sim_x_pos_min).mul_(self.pos_scale)
        y.sub_(self.sim_y_pos_min).mul_(self.pos_scale)
        assert paddle.all(0 <= x), f"error in {sim_name} x.min={x._min().item()}"
        assert paddle.all(
            x < self.max_x_pos
        ), f"error in {sim_name} y.max={x._max().item()}"
        assert paddle.all(0 <= y), f"error in {sim_name} y.min={y._min().item()}"
        assert paddle.all(
            y < self.max_y_pos
        ), f"error in {sim_name} y.max={y._max().item()}"
        all_pos = paddle.stack([x, y], dim=1)
        return all_pos

    def getitem_all_pos(self, idx, ctx=None):
        if ctx is not None and "all_pos" in ctx:
            return ctx["all_pos"]
        sim_name = self._get_sim_name(idx)
        all_pos = self._load_xy(self.source_root / sim_name)
        if ctx is not None:
            assert "all_pos" not in ctx
            ctx["all_pos"] = all_pos
        return all_pos

    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is not None and "mesh_pos" in ctx:
            return ctx["mesh_pos"]
        mesh_pos = self.getitem_all_pos(idx, ctx=ctx)
        mesh_pos = self._downsample_input(mesh_pos, idx=idx, ctx=ctx)
        if ctx is not None:
            assert "mesh_pos" not in ctx
            ctx["mesh_pos"] = mesh_pos
        return mesh_pos

    def getitem_query_pos(self, idx, ctx=None):
        if ctx is not None and "query_pos" in ctx:
            return ctx["query_pos"]
        query_pos = self.getitem_all_pos(idx, ctx=ctx)
        query_pos = self._downsample_query(query_pos, idx=idx, ctx=ctx)
        if self.standardize_query_pos:
            query_pos = (
                query_pos
                / (paddle.tensor([self.max_x_pos, self.max_y_pos])[(None), :] / 2)
                - 1
            )
        if ctx is not None:
            assert "query_pos" not in ctx
            ctx["query_pos"] = query_pos
        return query_pos

    def getitem_reconstruction_pos(self, idx, ctx=None):
        if ctx is not None and "rec_pos" in ctx:
            return ctx["rec_pos"]
        rec_pos = self.getitem_all_pos(idx, ctx=ctx)
        rec_pos = self._downsample_reconstruction_output(rec_pos, idx=idx, ctx=ctx)
        if ctx is not None:
            assert "rec_pos" not in ctx
            ctx["rec_pos"] = rec_pos
        return rec_pos

    def getitem_grid_pos(self, idx=None, ctx=None):
        if ctx is not None and "grid_pos" in ctx:
            return ctx["grid_pos"]
        assert self.grid_resolution is not None
        x_linspace = paddle.linspace(0, self.max_x_pos, self.grid_resolution[0])
        y_linspace = paddle.linspace(0, self.max_y_pos, self.grid_resolution[1])
        grid_pos = (
            paddle.stack(paddle.meshgrid(x_linspace, y_linspace, indexing="ij"))
            .flatten(start_dim=1)
            .T
        )
        if ctx is not None:
            assert "grid_pos" not in ctx
            ctx["grid_pos"] = grid_pos
        return grid_pos

    def getitem_mesh_edges(self, idx, ctx=None):
        assert self.grid_resolution is None
        if self.radius_graph_r is None:
            return None
        sim_name = self._get_sim_name(idx)
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        if self.supernode_edge_mode == "mesh_to_mesh":
            if self.num_supernodes is None:
                flow = "source_to_target"
            else:
                flow = "target_to_source"
            edges = radius_graph(
                x=mesh_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
                flow=flow,
            )
            if self.num_supernodes is not None:
                generator = self._get_generator(idx)
                perm = paddle.randperm(len(mesh_pos))[: self.num_supernodes]
                is_supernode_edge = paddle.isin(edges[0], perm)
                edges = edges[:, (is_supernode_edge)]
        elif self.supernode_edge_mode == "mesh_to_supernode":
            assert self.num_supernodes is not None
            generator = self._get_generator(idx)
            perm = paddle.randperm(len(mesh_pos))[: self.num_supernodes]
            supernodes_pos = mesh_pos[perm]
            edges = radius(
                x=mesh_pos,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            edges[0] = perm[edges[0]]
        else:
            raise NotImplementedError
        return edges.T

    def getitem_mesh_to_grid_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.num_supernodes is None
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        if self.radius_graph_r is None:
            return None
        edges = radius(
            x=mesh_pos,
            y=grid_pos,
            r=self.radius_graph_r,
            max_num_neighbors=self.radius_graph_max_num_neighbors,
        ).T
        return edges

    def getitem_grid_to_query_edges(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.num_supernodes is None
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        query_pos = self.getitem_query_pos(idx, ctx=ctx)
        if self.radius_graph_r is None:
            return None
        edges = radius(
            x=grid_pos,
            y=query_pos,
            r=self.radius_graph_r,
            max_num_neighbors=self.radius_graph_max_num_neighbors,
        ).T
        return edges

    def getshape_x(self):
        sim_name, timestep_to_fname = self.samples[0]
        num_channels = paddle.load(
            path=str(self.source_root / sim_name / timestep_to_fname[0])
        ).T.size(1)
        return None, num_channels * self.num_input_timesteps

    def getshape_target(self):
        sim_name, timestep_to_fname = self.samples[0]
        num_channels = paddle.load(
            path=str(self.source_root / sim_name / timestep_to_fname[0])
        ).T.size(1)
        return None, num_channels

    def getitem_target_t0(self, idx, ctx=None):
        assert self.num_input_timesteps == float("inf")
        sim_name, timestep_to_fname = self.samples[idx]
        data = paddle.load(
            path=str(self.source_root / sim_name / timestep_to_fname[0])
        ).T
        data = self._downsample_query(data, idx=idx, ctx=ctx)
        data -= self.mean.view(1, -1)
        data /= self.std.view(1, -1)
        data = data.float()
        data = self._clamp(data)
        return data

    def getitem_target(self, idx, ctx=None):
        if self.num_input_timesteps == float("inf"):
            sim_name, timestep_to_fname = self.samples[idx]
            data = [
                paddle.load(
                    path=str(self.source_root / sim_name / timestep_to_fname[i])
                ).T
                for i in range(1, self.max_timestep + 1)
            ]
            data = paddle.stack(
                [self._downsample_query(item, idx=idx, ctx=ctx) for item in data], dim=2
            )
            data -= self.mean.view(1, -1, 1)
            data /= self.std.view(1, -1, 1)
        else:
            seqidx = idx // self.max_timestep
            sim_name, timestep_to_fname = self.samples[seqidx]
            timestep = self.getitem_timestep(idx, ctx=ctx)
            data = paddle.load(
                path=str(self.source_root / sim_name / timestep_to_fname[timestep + 1])
            ).T
            data = self._downsample_query(data, idx=idx, ctx=ctx)
            data -= self.mean.view(1, -1)
            data /= self.std.view(1, -1)
        data = data.float()
        data = self._clamp(data)
        return data

    def getitem_reconstruction_input(self, idx, ctx=None):
        assert self.num_input_timesteps != float("inf")
        seqidx = idx // self.max_timestep
        sim_name, timestep_to_fname = self.samples[seqidx]
        timestep = self.getitem_timestep(idx, ctx=ctx)
        data = paddle.load(
            path=str(self.source_root / sim_name / timestep_to_fname[timestep + 1])
        ).T
        data = self._downsample_input(data, idx=idx, ctx=ctx)
        data -= self.mean.view(1, -1)
        data /= self.std.view(1, -1)
        data = data.float()
        data = self._clamp(data)
        return data

    def getitem_reconstruction_output(self, idx, ctx=None):
        assert self.num_input_timesteps != float("inf")
        seqidx = idx // self.max_timestep
        sim_name, timestep_to_fname = self.samples[seqidx]
        timestep = self.getitem_timestep(idx, ctx=ctx)
        data = paddle.load(
            path=str(self.source_root / sim_name / timestep_to_fname[timestep + 1])
        ).T
        data = self._downsample_reconstruction_output(data, idx=idx, ctx=ctx)
        data -= self.mean.view(1, -1)
        data /= self.std.view(1, -1)
        data = data.float()
        data = self._clamp(data)
        return data

    def _clamp(self, data):
        if self.clamp is not None:
            if self.clamp_mode == "hard":
                data = data.clamp(-self.clamp, self.clamp)
            elif self.clamp_mode == "log":
                apply = data.abs() > self.clamp
                values = data[apply]
                data[apply] = paddle.sign(values) * (
                    self.clamp + paddle.log(1 + values.abs()) - np.log(1 + self.clamp)
                )
            else:
                raise NotImplementedError
        return data

    def getitem_x(self, idx, ctx=None):
        if self.num_input_timesteps == float("inf"):
            sim_name, timestep_to_fname = self.samples[idx]
            data = paddle.load(
                path=str(self.source_root / sim_name / timestep_to_fname[0])
            ).T
            data = self._downsample_input(data, idx=idx, ctx=ctx)
            data -= self.mean.view(1, -1)
            data /= self.std.view(1, -1)
        else:
            seqidx = idx // self.max_timestep
            sim_name, timestep_to_fname = self.samples[seqidx]
            timestep = self.getitem_timestep(idx, ctx=ctx)
            data = paddle.stack(
                [
                    paddle.load(
                        path=str(
                            self.source_root / sim_name / timestep_to_fname[max(0, i)]
                        )
                    ).T
                    for i in range(
                        timestep - self.num_input_timesteps + 1, timestep + 1
                    )
                ],
                dim=1,
            )
            data = self._downsample_input(data, idx=idx, ctx=ctx)
            data -= self.mean.view(1, 1, -1)
            data /= self.std.view(1, 1, -1)
            data = einops.rearrange(
                data,
                "num_points timesteps num_channels -> num_points (timesteps num_channels)",
            )
        data = data.float()
        data = self._clamp(data)
        return data

    def getitem_interpolated(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.standardize_query_pos
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        x_linspace = paddle.linspace(0, self.max_x_pos, self.grid_resolution[1])
        y_linspace = paddle.linspace(0, self.max_y_pos, self.grid_resolution[0])
        grid_pos = paddle.meshgrid(x_linspace, y_linspace, indexing="xy")
        x = self.getitem_x(idx, ctx=ctx)
        grid = paddle.from_numpy(
            scipy.interpolate.griddata(
                mesh_pos.unbind(1), x, grid_pos, method="linear", fill_value=0.0
            )
        ).float()
        return grid
