import sys

sys.path.append("/home/ggbond/baidu_98/paddle_project")
import os
from argparse import ArgumentParser
from pathlib import Path

import einops
import paddle
from paddle_utils import *
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="e.g. /system/user/publicdata/CVSim/mesh_dataset/v1",
    )
    parser.add_argument("--q", type=float, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--exclude_last", type=int, default=0)
    return vars(parser.parse_args())


def get_paddle_files(root):
    result = []
    for fname in os.listdir(root):
        uri = root / fname
        if uri.is_dir():
            result += get_paddle_files(uri)
        elif (
            uri.name.endswith(".th")
            and not uri.name.startswith("coordinates")
            and not uri.name.startswith("geometry2d")
            and not uri.name.startswith("object_mask")
            and not uri.name.startswith("U_init")
            and not uri.name.startswith("num_objects")
            and not uri.name.startswith("x")
            and not uri.name.startswith("y")
            and not uri.name.startswith("edge_index")
            and not uri.name.startswith("movement_per_position")
            and not uri.name.startswith("sampling_weights")
        ):
            try:
                _ = int(uri.name[: len("00000000")])
            except:
                print(f"{uri.name} is not a data file")
                raise
            result.append(uri)
    return result


class MeanVarDataset(paddle.io.Dataset):
    def __init__(self, case_uris, q):
        super().__init__()
        self.case_uris = case_uris
        self.q = q

    def __len__(self):
        return len(self.case_uris)

    def __getitem__(self, idx):
        case_uri = self.case_uris[idx]
        assert case_uri.name.startswith("case_")
        uris = get_paddle_files(case_uri)
        if len(uris) != 120:
            raise RuntimeError(
                f"invalid number of uris for case '{case_uri.as_posix()}' len={len(uris)}"
            )
        data = paddle.stack([paddle.load(path=str(uri)) for uri in uris])
        mean = paddle.zeros(3)
        var = paddle.zeros(3)
        mmin = paddle.zeros(3)
        mmax = paddle.zeros(3)
        within1std = paddle.zeros(3)
        within2std = paddle.zeros(3)
        within3std = paddle.zeros(3)
        for i in range(3):
            cur_data = data[:, (i)]
            if self.q > 0:
                cur_mean = cur_data.mean()
                cur_std = cur_data.std()
                dist = paddle.distribution.Normal(loc=0, scale=1)
                z_qmin = paddle.sqrt(paddle.to_tensor(2.0)) * paddle.erfinv(2 * paddle.tensor(self.q) - 1)
                z_qmax = paddle.sqrt(paddle.to_tensor(2.0)) * paddle.erfinv(2 * paddle.tensor(1 - self.q) - 1)
                qmin = cur_mean + cur_std * z_qmin
                qmax = cur_mean + cur_std * z_qmax
                is_valid = paddle.logical_and(qmin < cur_data, cur_data < qmax)
                valid_data = cur_data[is_valid]
            else:
                valid_data = cur_data
            mean[i] = valid_data.mean()
            var[i] = valid_data.var()
            mmin[i] = valid_data._min()
            mmax[i] = valid_data._max()
            cur_std = valid_data.std()
            is_within1std = paddle.logical_and(
                mean[i] - 1 * cur_std < valid_data, valid_data < mean[i] + 1 * cur_std
            )
            within1std[i] = is_within1std.sum() / is_within1std.size
            is_within2std = paddle.logical_and(
                mean[i] - 2 * cur_std < valid_data, valid_data < mean[i] + 2 * cur_std
            )
            within2std[i] = is_within2std.sum() / is_within2std.size
            is_within3std = paddle.logical_and(
                mean[i] - 3 * cur_std < valid_data, valid_data < mean[i] + 3 * cur_std
            )
            within3std[i] = is_within3std.sum() / is_within3std.size
        return mean, var, mmin, mmax, within1std, within2std, within3std


def main(root, num_workers, exclude_last, q):
    root = Path(root).expanduser()
    assert root.exists() and root.is_dir()
    print(f"root: {root.as_posix()}")
    print(f"num_workers: {num_workers}")
    print(f"exclude_last: {exclude_last}")
    assert q < 0.5
    print(f"q (exclude values below/above quantile): {q}")
    case_uris = [(root / fname) for fname in os.listdir(root)]
    case_uris = list(
        sorted(case_uris, key=lambda cu: int(cu.name.replace("case_", "")))
    )
    if exclude_last > 0:
        case_uris = case_uris[:-exclude_last]
    print(f"using {len(case_uris)} uris")
    print(f"last used case_uri: {case_uris[-1].as_posix()}")
    dataset = MeanVarDataset(case_uris=case_uris, q=q)
    sum_of_means = 0.0
    sum_of_vars = 0.0
    min_of_mins = paddle.full(size=(3,), fill_value=paddle.inf)
    max_of_maxs = paddle.full(size=(3,), fill_value=-paddle.inf)
    within1std_sum = paddle.zeros(3)
    within2std_sum = paddle.zeros(3)
    within3std_sum = paddle.zeros(3)
    for data in tqdm(
        paddle.io.DataLoader(dataset=dataset, batch_size=1, num_workers=num_workers)
    ):
        mean, var, mmin, mmax, within1std, within2std, within3std = data
        sum_of_means += mean.squeeze(0)
        sum_of_vars += var.squeeze(0)
        min_of_mins = paddle.minimum(min_of_mins, mmin.squeeze(0))
        max_of_maxs = paddle.maximum(max_of_maxs, mmax.squeeze(0))
        within1std_sum += within1std.squeeze(0)
        within2std_sum += within2std.squeeze(0)
        within3std_sum += within3std.squeeze(0)
    mean = sum_of_means / len(dataset)
    std = paddle.sqrt(sum_of_vars / len(dataset))
    within1std_mean = within1std_sum / len(dataset)
    within2std_mean = within2std_sum / len(dataset)
    within3std_mean = within3std_sum / len(dataset)
    print(f"data_mean: {mean.tolist()}")
    print(f"data_std: {std.tolist()}")
    print(f"data_min: {min_of_mins.tolist()}")
    print(f"data_max: {max_of_maxs.tolist()}")
    print(f"within1std: {within1std_mean.tolist()}")
    print(f"within2std: {within2std_mean.tolist()}")
    print(f"within3std: {within3std_mean.tolist()}")


if __name__ == "__main__":
    main(**parse_args())
