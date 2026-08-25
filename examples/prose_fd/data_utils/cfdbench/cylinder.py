import random
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import paddle
from tqdm import tqdm

from .base import CfdAutoDataset, CfdDataset
from .utils import load_json, normalize_bc, normalize_physics_props


def load_case_data(case_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load from the file that I have preprocessed, and pad the boundary conditions,
    turn into a numpy array of features.

    The shape of both u and v is (time steps, height, width)
    """
    case_params = load_json(case_dir / "case.json")
    u_file = case_dir / "u.npy"
    v_file = case_dir / "v.npy"
    u = np.load(u_file)
    v = np.load(v_file)
    mask = np.ones_like(u)
    x_min = case_params["x_min"]
    x_max = case_params["x_max"]
    y_min = case_params["y_min"]
    y_max = case_params["y_max"]
    radius = case_params["radius"]
    case_params["center_x"] = -x_min
    case_params["center_y"] = -y_min
    for key in ["x_min", "x_max", "y_min", "y_max"]:
        del case_params[key]
    height = y_max - y_min
    width = x_max - x_min
    case_params["height"] = height
    case_params["width"] = width
    dx = width / u.shape[2]
    dy = height / u.shape[1]
    for i in range(u.shape[1]):
        for j in range(u.shape[2]):
            x = x_min + j * dx
            y = y_min + i * dy
            if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= radius**2:
                mask[:, i, j] = 0
    u = np.pad(
        u,
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=case_params["vel_in"],
    )
    v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    features = np.stack([u, v, mask], axis=1)
    return features, case_params


class CylinderFlowDataset(CfdDataset):
    """
    Dataset for Cylinder flow problem.

    Varying density and viscosity and inlet velocity for each case (3 variables).
    """

    data_delta_time = 0.1
    data_max_time = 30
    case_params_keys = [
        "vel_in",
        "density",
        "viscosity",
        "height",
        "width",
        "center_x",
        "center_y",
        "radius",
    ]

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        sample_point_by_point: bool = False,
        stable_state_diff: float = 0.001,
    ):
        """
        Args:
        - data_dir:
        - norm_props: whether to normalize physics properties.
        - sample_point_by_point: If True, each example is a feature point
            (x, y, t) and the corresponding output function value u(x, y, t).
            If False, each example is an entire frame.
        - stable_state_diff: The mean relative difference between two consecutive
            frames that indicates the system has reached a stable state.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.sample_point_by_point = sample_point_by_point
        self.stable_state_diff = stable_state_diff
        self.load_data(case_dirs)

    def load_data(self, case_dirs: List[Path]):
        """
        This will set the following attributes:
            self.case_params: List[dict]
            self.features: List[Tensor]  # (N, T, 2, h, w)
            self.case_ids: List[int]  # Each sample's case ID
        where N is the number of cases.
        """
        self.case_params: List[paddle.Tensor] = []
        self.num_features = 0
        self.num_frames: List[int] = []
        features: List[paddle.Tensor] = []
        case_ids: List[int] = []
        for case_id, case_dir in enumerate(tqdm(case_dirs)):
            this_case_features, this_case_params = load_case_data(case_dir)
            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "vel_in")
            T, c, h, w = this_case_features.shape
            self.num_features += T * h * w
            params_tensor = paddle.tensor(
                [this_case_params[key] for key in self.case_params_keys],
                dtype=paddle.float32,
            )
            self.case_params.append(params_tensor)
            features.append(paddle.tensor(this_case_features, dtype=paddle.float32))
            case_ids.append(case_id)
            self.num_frames.append(T)
        self.features = features
        self.case_ids = paddle.tensor(case_ids)
        self.num_frames_before: List[int] = [
            sum(self.num_frames[: i + 1]) for i in range(len(self.num_frames))
        ]

    def idx_to_case_id_and_frame_idx(self, idx: int) -> Tuple[int, int]:
        """
        Given an index, return the case ID of the corresponding example.
        Will be using `self.num_frames_before`.

        For instance, if the number of frames in the first three cases are
        [10, 12, 11], then:
        - 0~9 should map to case_id = 0
        - 10~21 should map to case_id = 1
        - 22~32 should map to case_id = 2
        In this case, `num_frames_before` should be [10, 22, 33].
        """
        case_id = bisect_right(self.num_frames_before, idx)
        if case_id == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.num_frames_before[case_id - 1]
        return case_id, frame_idx

    def __getitem__(self, idx: int):
        case_id, frame_idx = self.idx_to_case_id_and_frame_idx(idx)
        t = paddle.tensor([frame_idx]).float()
        frame = self.features[case_id][frame_idx]
        case_params = self.case_params[case_id]
        return case_params, t, frame

    def __len__(self):
        return self.num_frames_before[-1]


class CylinderFlowAutoDataset(CfdAutoDataset):
    """
    Dataset for Laminar flow problem.

    Varying density and viscosity and inlet velocity for each case (3 variables).
    """

    data_delta_time = 0.001

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        split: str,
        cache_dir: Path,
        delta_time: float = 0.1,
        stable_state_diff: float = 0.001,
    ):
        """
        Assume:
        - time: 30s
        - time steps: 120
        - time step size: 0.25s

        geometry 0：d=0.1m，l=1m
        geometry 1-10 ：d=0.05-0.09m,0.11-0.15m，l=1m
        geometry 11-20 ：d=0.1m，l=0.5-0.95m

        case0：入口速度0.1 m/s，密度1000 kg /m^3，动力粘度0.01 Pa-s
        case1-case21：入口速度0.05-0.15 m/s，du=0.005 m/s
        case22-case42：密度900-1100 kg /m^3，dρ=10 kg/m^3
        case43-case63：动力粘度0.005-0.015 Pa-s，dv=0.0005 Pa-s

        Args:
            data_dir: Path to the data directory.
            delta_time: Time step size (in sec) to use for training.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.split = split
        self.cache_dir = cache_dir / split
        self.delta_time = delta_time
        self.stable_state_diff = stable_state_diff
        self.time_step_size = int(self.delta_time / self.data_delta_time)
        self.load_data(case_dirs, self.time_step_size)

    def load_data(self, case_dirs, time_step_size: int):
        """
        This will set the following attributes:
            self.case_dirs: List[Path]
            self.case_params: List[dict]
            self.inputs: List[Tensor]  # (2, h, w)
            self.labels: List[Tensor]  # (2, h, w)
            self.case_ids: List[int]  # Each sample's case ID
        """
        if self.cache_dir.exists():
            print(f"Loading from cache: {self.cache_dir}")
            self.inputs = paddle.load(path=str(self.cache_dir / "inputs.pt"))
            self.labels = paddle.load(path=str(self.cache_dir / "labels.pt"))
            self.case_ids = paddle.load(path=str(self.cache_dir / "case_ids.pt"))
            self.case_params = paddle.load(path=str(self.cache_dir / "case_params.pt"))
            self.all_features = paddle.load(
                path=str(self.cache_dir / "all_features.pt")
            )
            return
        self.case_params: List[dict] = []
        all_inputs: List[paddle.Tensor] = []
        all_labels: List[paddle.Tensor] = []
        all_case_ids: List[int] = []
        self.all_features: List[paddle.Tensor] = []
        for case_id, case_dir in enumerate(case_dirs):
            case_features, this_case_params = load_case_data(case_dir)
            self.all_features.append(case_features)
            inputs = case_features[:-time_step_size, :]
            outputs = case_features[time_step_size:, :]
            assert len(inputs) == len(outputs)
            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "vel_in")
            self.case_params.append(this_case_params)
            num_steps = len(outputs)
            for i in range(num_steps):
                inp = paddle.tensor(inputs[i], dtype=paddle.float32)
                out = paddle.tensor(outputs[i], dtype=paddle.float32)
                inp_magn = paddle.sqrt(inp[0] ** 2 + inp[1] ** 2)
                out_magn = paddle.sqrt(out[0] ** 2 + out[1] ** 2)
                diff = paddle.abs(inp_magn - out_magn).mean()
                if diff < self.stable_state_diff:
                    print(f"Converged at {i} out of {num_steps}, {this_case_params}")
                    break
                assert not paddle.isnan(inp).any()
                assert not paddle.isnan(out).any()
                all_inputs.append(inp)
                all_labels.append(out)
                all_case_ids.append(case_id)
        self.inputs = paddle.stack(all_inputs)
        self.labels = paddle.stack(all_labels)
        self.case_ids = all_case_ids
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        paddle.save(obj=self.inputs, path=self.cache_dir / "inputs.pt")
        paddle.save(obj=self.labels, path=self.cache_dir / "labels.pt")
        paddle.save(obj=self.case_ids, path=self.cache_dir / "case_ids.pt")
        paddle.save(obj=self.case_params, path=self.cache_dir / "case_params.pt")
        paddle.save(obj=self.all_features, path=self.cache_dir / "all_features.pt")

    def __getitem__(self, idx: int):
        inputs = self.inputs[idx]
        label = self.labels[idx]
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id]
        case_params = {
            k: paddle.tensor(v, dtype=paddle.float32) for k, v in case_params.items()
        }
        return inputs, label, case_params

    def __len__(self):
        return len(self.inputs)


def get_cylinder_datasets(
    data_dir: Path, case_name: str, norm_props: bool, norm_bc: bool, seed: int = 0
) -> Tuple[CfdDataset, CfdDataset, CfdDataset]:
    print(data_dir, case_name)
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in case_name:
            case_dir = data_dir / name
            this_case_dirs = sorted(
                case_dir.glob("case*"), key=lambda x: int(x.name[4:])
            )
            case_dirs += this_case_dirs
    assert case_dirs
    random.seed(seed)
    random.shuffle(case_dirs)
    num_cases = len(case_dirs)
    num_train = int(num_cases * 0.8)
    num_dev = int(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    print("==== Number of cases in different splits ====")
    print(
        f"train: {len(train_case_dirs)}, dev: {len(dev_case_dirs)}, test: {len(test_case_dirs)}"
    )
    print("=============================================")
    kwargs: dict[str, Any] = dict(norm_props=norm_props, norm_bc=norm_bc)
    train_data = CylinderFlowDataset(train_case_dirs, **kwargs)
    dev_data = CylinderFlowDataset(dev_case_dirs, **kwargs)
    test_data = CylinderFlowDataset(test_case_dirs, **kwargs)
    return train_data, dev_data, test_data


def get_cylinder_auto_datasets(
    data_dir: Path,
    subset_name: str,
    norm_props: bool,
    norm_bc: bool,
    delta_time: float = 0.01,
    stable_state_diff: float = 0.001,
    seed: int = 0,
    load_splits: List[str] = ["train", "dev", "test"],
) -> Tuple[CylinderFlowAutoDataset, CylinderFlowAutoDataset, CylinderFlowAutoDataset]:
    print(data_dir, subset_name)
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in subset_name:
            case_dir = data_dir / name
            this_case_dirs = sorted(
                case_dir.glob("case*"), key=lambda x: int(x.name[4:])
            )
            case_dirs += this_case_dirs
    assert case_dirs
    random.seed(seed)
    random.shuffle(case_dirs)
    num_cases = len(case_dirs)
    num_train = int(num_cases * 0.8)
    num_dev = int(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    print("==== Number of cases in different splits ====")
    print(
        f"train: {len(train_case_dirs)}, dev: {len(dev_case_dirs)}, test: {len(test_case_dirs)}"
    )
    print("=============================================")
    kwargs: dict[str, Any] = dict(
        delta_time=delta_time,
        stable_state_diff=stable_state_diff,
        norm_props=norm_props,
        norm_bc=norm_bc,
        cache_dir=Path("./dataset/cache/cylinder", subset_name),
    )
    if "train" in load_splits:
        train_data = CylinderFlowAutoDataset(train_case_dirs, split="train", **kwargs)
    else:
        train_data = None
    if "dev" in load_splits:
        dev_data = CylinderFlowAutoDataset(dev_case_dirs, split="dev", **kwargs)
    else:
        dev_data = None
    if "test" in load_splits:
        test_data = CylinderFlowAutoDataset(test_case_dirs, split="test", **kwargs)
    else:
        test_data = None
    return train_data, dev_data, test_data


if __name__ == "__main__":
    data_dir = Path("../../data/large/cfdbench/cylinder")
    time_step_size = 10
    dataset = CylinderFlowDataset([data_dir], norm_props=True, norm_bc=True)
    exit()
