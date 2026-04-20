import os

import paddle
from ppcfd.models.g_fno import FNO2d, FNO3d
from ppcfd.models.g_fno import GCNN2d, GCNN3d
from ppcfd.models.g_fno import GFNO2d, GFNO3d
from ppcfd.models.g_fno import Ghybrid2d
from ppcfd.models.g_fno import radialNO2d, radialNO3d
from ppcfd.models.g_fno.paddle_utils import _set_num_threads
from ppcfd.models.g_fno.paddle_utils import move_to_device
from ppcfd.models.g_fno.paddle_utils import set_runtime_device

"""
This is a modified version of fourier_2d_time.py from https://github.com/zongyi-li/fourier_neural_operator
"""
import datetime
import random
import argparse
from timeit import default_timer

import h5py
import numpy as np
import scipy
import xarray as xr
from tqdm import tqdm
from utils import LpLoss, eq_check_rf, eq_check_rt, pde_data

_set_num_threads(1)


def get_eval_pred(model, x, strategy, T, times):
    if strategy == "oneshot":
        pred = model(x)
    else:
        for t in range(T):
            t1 = default_timer()
            im = model(x)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = paddle.cat((pred, im), -2)
            if strategy == "markov":
                x = im
            else:
                x = paddle.cat((x[..., 1:, :], im), dim=-2)
    return pred


parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_path", type=str, default="./results/tmp", help="path to store results"
)
parser.add_argument(
    "--suffix", type=str, default=None, help="suffix to add to the results path"
)
parser.add_argument(
    "--txt_suffix", type=str, default=None, help="suffix to add to the results txt"
)
parser.add_argument("--data_path", type=str, required=True, help="path to the data")
parser.add_argument(
    "--super_path", type=str, default=None, help="path to the superresolution data"
)
parser.add_argument("--super", action="store_true", help="enable superres testing")
parser.add_argument("--verbose", action="store_true")
parser.add_argument(
    "--T", type=int, required=True, help="number of timesteps to predict"
)
parser.add_argument("--ntrain", type=int, required=True, help="training sample size")
parser.add_argument("--nvalid", type=int, required=True, help="valid sample size")
parser.add_argument("--ntest", type=int, required=True, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=20)
parser.add_argument(
    "--Gwidth",
    type=int,
    default=10,
    help="hidden dimension of equivariant layers if model_type=hybrid",
)
parser.add_argument(
    "--n_equiv",
    type=int,
    default=3,
    help="number of equivariant layers if model_type=hybrid",
)
parser.add_argument(
    "--reflection",
    action="store_true",
    help="symmetry group p4->p4m for data augmentation",
)
parser.add_argument(
    "--grid", type=str, default=None, help="[symmetric, cartesian, None]"
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument(
    "--early_stopping",
    type=int,
    default=None,
    help="stop if validation error does not improve for successive epochs",
)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument(
    "--gamma", type=float, default=None, help="gamma for step scheduler"
)
parser.add_argument(
    "--step_size", type=int, default=None, help="step size for step scheduler"
)
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument(
    "--strategy", type=str, default="markov", help="markov, recurrent or oneshot"
)
parser.add_argument(
    "--time_pad",
    action="store_true",
    help="pad the time dimension for strategy=oneshot",
)
parser.add_argument(
    "--noise_std",
    type=float,
    default=0.0,
    help="amount of noise to inject for strategy=markov",
)
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="runtime device, e.g. auto, cpu, gpu, xpu, npu, gcu",
)
parser.add_argument(
    "--rdb_super_res",
    type=int,
    default=128,
    help="original spatial resolution for PDEBench radial dam break data",
)
parser.add_argument(
    "--rdb_downsample",
    type=int,
    default=4,
    help="spatial downsampling factor for PDEBench radial dam break data",
)
args = parser.parse_args()
assert args.model_type in [
    "FNO2d",
    "FNO2d_aug",
    "FNO3d",
    "FNO3d_aug",
    "GCNN2d_p4",
    "GCNN2d_p4m",
    "GCNN3d_p4",
    "GCNN3d_p4m",
    "GFNO2d_p4",
    "GFNO2d_p4m",
    "GFNO3d_p4",
    "GFNO3d_p4m",
    "Ghybrid2d_p4",
    "Ghybrid2d_p4m",
    "radialNO2d_p4",
    "radialNO2d_p4m",
    "radialNO3d_p4",
    "radialNO3d_p4m",
], f"Invalid model type {args.model_type}"
assert args.strategy in [
    "teacher_forcing",
    "markov",
    "recurrent",
    "oneshot",
], "Invalid training strategy"
runtime_device = set_runtime_device(args.device)
paddle.seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def to_runtime(obj):
    return move_to_device(obj, runtime_device)


def move_batch(xx, yy):
    return to_runtime(xx), to_runtime(yy)


data_aug = "aug" in args.model_type
TRAIN_PATH = args.data_path
S = Sx = Sy = 64
S_super = 4 * S
T_in = 10
T = args.T
T_super = 4 * T
d = 2
num_channels = 1
threeD = args.model_type in [
    "FNO3d",
    "FNO3d_aug",
    "GCNN3d_p4",
    "GCNN3d_p4m",
    "GFNO3d_p4",
    "GFNO3d_p4m",
    "radialNO3d_p4",
    "radialNO3d_p4m",
]
extension = TRAIN_PATH.split(".")[-1]
swe = os.path.split(TRAIN_PATH)[-1] == "ShallowWater2D"
rdb = TRAIN_PATH.split(os.path.sep)[-1][:6] == "2D_rdb"
grid_type = "symmetric"
if args.grid:
    grid_type = args.grid
    assert grid_type in ["symmetric", "cartesian", "None"]
if rdb:
    assert T == 24, "T should be 24 for rdb"
    assert args.rdb_downsample > 0, "rdb_downsample should be positive"
    assert args.rdb_super_res > 0, "rdb_super_res should be positive"
    assert (
        args.rdb_super_res % args.rdb_downsample == 0
    ), "rdb_super_res should be divisible by rdb_downsample"
    T_in = 1
    S_super = args.rdb_super_res
    S = Sx = Sy = S_super // args.rdb_downsample
    T_super = 96
elif swe:
    assert not args.super, "Superresolution not supported for pdearena"
    assert T == 9, "T should be 9 for swe"
    T_in = 2
    Sy, Sx = 96, 192
    num_channels = 2
    grid_type = "cartesian"
spatial_dims = range(1, d + 1)
if args.strategy == "oneshot":
    assert threeD, "oneshot strategy only for 3d models"
if threeD:
    assert args.strategy == "oneshot", "threeD models use oneshot strategy"
ntrain = args.ntrain
nvalid = args.nvalid
ntest = args.ntest
time_modes = None
time = args.strategy == "oneshot"
if time and not args.time_pad:
    time_modes = 5 if swe else 6
elif time and swe:
    time_modes = 8
modes = args.modes
width = args.width
n_layer = args.depth
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
scheduler_step = args.step_size
scheduler_gamma = args.gamma
initial_step = 1 if args.strategy == "markov" else T_in
root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
if args.suffix:
    root += "_" + args.suffix
os.makedirs(root)
path_model = os.path.join(root, "model.pt")
if args.model_type in ["FNO2d", "FNO2d_aug"]:
    model = to_runtime(
        FNO2d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes1=modes,
            modes2=modes,
            width=width,
            grid_type=grid_type,
        )
    )
elif args.model_type in ["FNO3d", "FNO3d_aug"]:
    modes3 = time_modes if time_modes else modes
    model = to_runtime(
        FNO3d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes1=modes,
            modes2=modes,
            modes3=modes3,
            width=width,
            time=time,
            time_pad=args.time_pad,
        )
    )
elif "GCNN2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        GCNN2d(
            num_channels=num_channels,
            initial_step=initial_step,
            width=width,
            reflection=reflection,
        )
    )
elif "GCNN3d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        GCNN3d(
            num_channels=num_channels,
            initial_step=initial_step,
            width=width,
            reflection=reflection,
        )
    )
elif "GFNO2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        GFNO2d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes=modes,
            width=width,
            reflection=reflection,
            grid_type=grid_type,
        )
    )
elif "GFNO3d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        GFNO3d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes=modes,
            time_modes=time_modes,
            width=width,
            reflection=reflection,
            grid_type=grid_type,
            time_pad=args.time_pad,
        )
    )
elif "Ghybrid2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        Ghybrid2d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes=modes,
            Gwidth=args.Gwidth,
            width=width,
            reflection=reflection,
            n_equiv=args.n_equiv,
        )
    )
elif "radialNO2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        radialNO2d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes=modes,
            width=width,
            reflection=reflection,
            grid_type=grid_type,
        )
    )
elif "radialNO3d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = to_runtime(
        radialNO3d(
            num_channels=num_channels,
            initial_step=initial_step,
            modes=modes,
            time_modes=time_modes,
            width=width,
            reflection=reflection,
            grid_type=grid_type,
            time_pad=args.time_pad,
        )
    )
else:
    raise NotImplementedError("Model not recognized")
if args.strategy == "oneshot":
    x_shape = [batch_size, Sy, Sx, T, initial_step, num_channels]
    x_shape_super = [1, S_super, S_super, T_super, initial_step, num_channels]
elif args.strategy == "markov":
    x_shape = [batch_size, Sy, Sx, 1, num_channels]
    x_shape_super = [1, *((S_super,) * d), 1, num_channels]
else:
    x_shape = [batch_size, Sy, Sx, T_in, num_channels]
    x_shape_super = [1, *((S_super,) * d), T_in, num_channels]
model.train()
x = to_runtime(paddle.randn(*x_shape))
if args.strategy == "recurrent":
    for _ in range(T):
        im = model(x)
        x = paddle.cat([x[..., 1:, :], im], dim=-2)
else:
    model(x)
eq_check_rt(model, x, spatial_dims)
eq_check_rf(model, x, spatial_dims)
if args.super:
    model.eval()
    with paddle.no_grad():
        x = to_runtime(paddle.randn(*x_shape_super))
        model(x)
full_data = None
if extension == "mat":
    assert num_channels == 1, "num channels should be 1 for .mat data"
    assert d == 2, "spatial dim should be 2 for .mat data"
    sub = 1
    try:
        with h5py.File(TRAIN_PATH, "r") as f:
            data = np.array(f["u"])
        data = np.transpose(data, axes=range(len(data.shape) - 1, -1, -1))
    except:
        data = scipy.io.loadmat(os.path.expandvars(TRAIN_PATH))["u"].astype(np.float32)
    data = data[..., None]
elif rdb:
    assert num_channels == 1, "num channels should be 1 for shallow water equations"
    assert d == 2, "spatial dim should be 2 for shallow water equations"
    with h5py.File(TRAIN_PATH, "r") as f:
        data_list = sorted(f.keys())
        data = np.concatenate(
            [np.array(f[key]["data"])[None] for key in data_list]
        ).transpose(0, 2, 3, 1, 4)[..., :-1, :]
        full_data = data[-ntest:]
        sampler = paddle.compat.nn.AvgPool2d(kernel_size=args.rdb_downsample)
        data = (
            sampler(paddle.tensor(data[..., ::4, 0]).permute(0, 3, 1, 2))
            .permute(0, 2, 3, 1)
            .unsqueeze(-1)
            .numpy()
        )
elif swe:
    assert num_channels == 2, "num channels should be 2 for shallow water equations"
    assert (
        ntrain + nvalid + ntest <= 5600 + 1120 + 1120
    ), f"Only {5600 + 1120 + 1120} solutions available"
    splits = {"train": ntrain, "valid": nvalid, "test": ntest}
    datas = {}
    for split, n in splits.items():
        if args.verbose:
            print(f"SWE: loading {split}")
        path = os.path.join(TRAIN_PATH, f"{split}.zarr")
        data = xr.open_zarr(path)
        normstat = paddle.load(path=str(os.path.join(TRAIN_PATH, "normstats.pt")))
        sample_rate = 8
        VORT_IND = 0
        PRES_IND = 1
        datas[split] = []
        for idx in tqdm(range(n), disable=not args.verbose):
            vort = paddle.tensor(data["vor"][idx].to_numpy())
            vort = (vort - normstat["vor"]["mean"]) / normstat["vor"]["std"]
            pres = paddle.tensor(data["pres"][idx].to_numpy())
            pres = (pres - normstat["pres"]["mean"]) / normstat["pres"]["std"]
            pres = pres.unsqueeze(1)
            pres = pres[4::sample_rate]
            vort = vort[4::sample_rate]
            pres_vort = paddle.cat([vort, pres], dim=1).permute(2, 3, 0, 1).unsqueeze(0)
            datas[split].append(pres_vort)
        datas[split] = paddle.cat(datas[split])
    data = paddle.cat([datas["train"], datas["valid"], datas["test"]])
else:
    raise ValueError(f"Extension {extension} not recognized")
assert data.shape[-2] >= T + T_in, "not enough time"
if args.super:
    assert not swe, "Superresolution is not supported for the PDE Arena SWE"
    assert full_data is not None or args.super_path is not None, "missing super dataset"
if not swe:
    data = paddle.from_numpy(data)
assert len(data) >= ntrain + nvalid + ntest, f"not enough data; {len(data)}"
train = data[:ntrain]
assert len(train) == ntrain, "not enough training data"
test = data[-ntest:]
test_rt = test.rot90(axes=list(spatial_dims)[:2])
test_rf = test.flip(axis=(spatial_dims[0],))
assert len(test) == ntest, "not enough test data"
valid = data[-(ntest + nvalid) : -ntest]
assert len(valid) == nvalid, "not enough validation data"
if args.verbose:
    print(f"{args.model_type}: Train/valid/test data shape: ")
    print(train.shape)
    print(valid.shape)
    print(test.shape)
assert Sx == train.shape[-3], f"Spatial downsampling should give {Sx} grid points"
assert Sy == train.shape[-4], f"Spatial downsampling should give {Sy} grid points"
train_data = pde_data(
    train, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std
)
ntrain = len(train_data)
valid_data = pde_data(valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
nvalid = len(valid_data)
test_data = pde_data(test, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
test_rt_data = pde_data(
    test_rt, train=False, strategy=args.strategy, T_in=T_in, T_out=T
)
test_rf_data = pde_data(
    test_rf, train=False, strategy=args.strategy, T_in=T_in, T_out=T
)
ntest = len(test_data)
train_loader = paddle.io.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True
)
valid_loader = paddle.io.DataLoader(
    dataset=valid_data, batch_size=batch_size, shuffle=False
)
test_loader = paddle.io.DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=False
)
test_rt_loader = paddle.io.DataLoader(
    dataset=test_rt_data, batch_size=batch_size, shuffle=False
)
test_rf_loader = paddle.io.DataLoader(
    dataset=test_rf_data, batch_size=batch_size, shuffle=False
)
complex_ct = sum(par.size * (1 + par.is_complex()) for par in model.parameters())
real_ct = sum(par.size for par in model.parameters())
if args.verbose:
    print(
        f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}"
    )
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(), learning_rate=learning_rate, weight_decay=args.lmbda
)
if args.step:
    assert args.step_size is not None, "step_size is None"
    assert scheduler_gamma is not None, "gamma is None"
    tmp_lr = paddle.optimizer.lr.StepDecay(
        step_size=args.step_size,
        gamma=scheduler_gamma,
        learning_rate=optimizer.get_lr(),
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
else:
    num_training_steps = epochs * len(train_loader)
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=num_training_steps, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
lploss = LpLoss(size_average=False)
best_valid = float("inf")
x_train, y_train = next(iter(train_loader))
x, y = move_batch(x_train, y_train)
x_valid, y_valid = next(iter(valid_loader))
if args.verbose:
    print(f"{args.model_type}; Input shape: {x.shape}, Target shape: {y.shape}")
if args.strategy == "oneshot":
    assert x_train[0].shape == paddle.Size([Sy, Sx, T, T_in, num_channels]), x_train[
        0
    ].shape
    assert y_train[0].shape == paddle.Size([Sy, Sx, T, num_channels]), y_train[0].shape
    assert x_valid[0].shape == paddle.Size([Sy, Sx, T, T_in, num_channels]), x_valid[
        0
    ].shape
    assert y_valid[0].shape == paddle.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
elif args.strategy == "markov":
    assert x_train[0].shape == paddle.Size([Sy, Sx, 1, num_channels]), x_train[0].shape
    assert y_train[0].shape == paddle.Size([Sy, Sx, num_channels]), y_train[0].shape
    assert x_valid[0].shape == paddle.Size([Sy, Sx, 1, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == paddle.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
else:
    assert x_train[0].shape == paddle.Size([Sy, Sx, T_in, num_channels]), x_train[
        0
    ].shape
    assert x_valid[0].shape == paddle.Size([Sy, Sx, T_in, num_channels]), x_valid[
        0
    ].shape
    assert y_valid[0].shape == paddle.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
    if args.strategy == "recurrent":
        assert y_train[0].shape == paddle.Size([Sy, Sx, T, num_channels]), y_train[
            0
        ].shape
    else:
        assert y_train[0].shape == paddle.Size([Sy, Sx, num_channels]), y_train[0].shape
model.eval()
if args.verbose:
    print(
        f"{args.model_type} pre-train equivariance checks: Rotations - {eq_check_rt(model, x, spatial_dims)}, Reflections - {eq_check_rf(model, x, spatial_dims)}"
    )
start = default_timer()
if args.verbose:
    print("Training...")
train_times = []
eval_times = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = train_vort_l2 = train_pres_l2 = 0
    for xx, yy in tqdm(train_loader, disable=not args.verbose):
        loss = 0
        xx, yy = move_batch(xx, yy)
        if data_aug:
            for b in range(len(xx)):
                for j in range(len(spatial_dims)):
                    for l in range(j + 1, len(spatial_dims)):
                        k_rt = random.randint(0, 3)
                        if k_rt > 0:
                            if not swe:
                                dims = [spatial_dims[j] - 1, spatial_dims[l] - 1]
                                xx[b] = xx[b].rot90(axes=dims, k=k_rt)
                                yy[b] = yy[b].rot90(axes=dims, k=k_rt)
                            elif b == 0:
                                dims = [spatial_dims[j], spatial_dims[l]]
                                xx = xx.rot90(axes=dims, k=k_rt)
                                yy = yy.rot90(axes=dims, k=k_rt)
                    if args.reflection:
                        k_rf = random.randint(0, 1)
                        if k_rf == 1:
                            xx[b] = xx[b].flip(axis=(spatial_dims[j] - 1,))
                            yy[b] = yy[b].flip(axis=(spatial_dims[j] - 1,))
        if args.strategy == "recurrent":
            for t in range(yy.shape[-2]):
                y = yy[..., t, :]
                im = model(xx)
                loss += lploss(
                    im.reshape(len(im), -1, num_channels),
                    y.reshape(len(y), -1, num_channels),
                )
                xx = paddle.cat((xx[..., 1:, :], im), dim=-2)
            loss /= yy.shape[-2]
        else:
            im = model(xx)
            if args.strategy == "oneshot":
                im = im.squeeze(-1)
            loss = lploss(
                im.reshape(len(im), -1, num_channels),
                yy.reshape(len(yy), -1, num_channels),
            )
        train_l2 += loss.item()
        if swe:
            train_vort_l2 += lploss(
                im[..., VORT_IND].reshape(len(im), -1, 1),
                yy[..., VORT_IND].reshape(len(yy), -1, 1),
            ).item()
            train_pres_l2 += lploss(
                im[..., PRES_IND].reshape(len(im), -1, 1),
                yy[..., PRES_IND].reshape(len(yy), -1, 1),
            ).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.step:
            scheduler.step()
    if args.step:
        scheduler.step()
    train_times.append(default_timer() - t1)
    valid_l2 = valid_vort_l2 = valid_pres_l2 = 0
    valid_loss_by_channel = None
    with paddle.no_grad():
        model.eval()
        model(xx)
        for xx, yy in valid_loader:
            xx, yy = move_batch(xx, yy)
            pred = get_eval_pred(
                model=model, x=xx, strategy=args.strategy, T=T, times=eval_times
            ).view(len(xx), Sy, Sx, T, num_channels)
            valid_l2 += lploss(
                pred.reshape(len(pred), -1, num_channels),
                yy.reshape(len(yy), -1, num_channels),
            ).item()
            if swe:
                valid_vort_l2 += lploss(
                    pred[..., VORT_IND].reshape(len(pred), -1, 1),
                    yy[..., VORT_IND].reshape(len(yy), -1, 1),
                ).item()
                valid_pres_l2 += lploss(
                    pred[..., PRES_IND].reshape(len(pred), -1, 1),
                    yy[..., PRES_IND].reshape(len(yy), -1, 1),
                ).item()
    t2 = default_timer()
    if args.verbose:
        print(
            f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, valid: {valid_l2 / nvalid}"
        )
    if valid_l2 < best_valid:
        best_epoch = ep
        best_valid = valid_l2
        paddle.save(obj=model.state_dict(), path=path_model)
    if args.early_stopping:
        if ep - best_epoch > args.early_stopping:
            break
stop = default_timer()
train_time = stop - start
train_times = paddle.tensor(train_times).mean().item()
num_eval = len(eval_times)
eval_times = paddle.tensor(eval_times).mean().item()
model.eval()
if args.verbose:
    print(
        f"{args.model_type} post-train equivariance checks: Rotations - {eq_check_rt(model, xx, spatial_dims)}, Reflections - {eq_check_rf(model, xx, spatial_dims)}"
    )
model.load_state_dict(paddle.load(path=str(path_model)))
model.eval()
test_l2 = test_vort_l2 = test_pres_l2 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
with paddle.no_grad():
    for xx, yy in test_loader:
        xx, yy = move_batch(xx, yy)
        pred = get_eval_pred(
            model=model, x=xx, strategy=args.strategy, T=T, times=[]
        ).view(len(xx), Sy, Sx, T, num_channels)
        test_l2 += lploss(
            pred.reshape(len(pred), -1, num_channels),
            yy.reshape(len(yy), -1, num_channels),
        ).item()
        if swe:
            test_vort_l2 += lploss(
                pred[..., VORT_IND].reshape(len(pred), -1, 1),
                yy[..., VORT_IND].reshape(len(yy), -1, 1),
            ).item()
            test_pres_l2 += lploss(
                pred[..., PRES_IND].reshape(len(pred), -1, 1),
                yy[..., PRES_IND].reshape(len(yy), -1, 1),
            ).item()
        rotations_l2 += lploss(
            model(xx)
            .rot90(axes=list(spatial_dims)[:2])
            .reshape(len(pred), -1, num_channels),
            model(xx.rot90(axes=list(spatial_dims)[:2])).reshape(
                len(pred), -1, num_channels
            ),
        )
        reflections_l2 += lploss(
            model(xx)
            .flip(axis=(spatial_dims[0],))
            .reshape(len(pred), -1, num_channels),
            model(xx.flip(axis=(spatial_dims[0],))).reshape(
                len(pred), -1, num_channels
            ),
        )
    for xx, yy in test_rt_loader:
        xx, yy = move_batch(xx, yy)
        pred = get_eval_pred(
            model=model, x=xx, strategy=args.strategy, T=T, times=[]
        ).view(len(xx), Sy, Sx, T, num_channels)
        test_rt_l2 += lploss(
            pred.reshape(len(pred), -1, num_channels),
            yy.reshape(len(yy), -1, num_channels),
        ).item()
    for xx, yy in test_rf_loader:
        xx, yy = move_batch(xx, yy)
        pred = get_eval_pred(
            model=model, x=xx, strategy=args.strategy, T=T, times=[]
        ).view(len(xx), Sy, Sx, T, num_channels)
        test_rf_l2 += lploss(
            pred.reshape(len(pred), -1, num_channels),
            yy.reshape(len(yy), -1, num_channels),
        ).item()
    rotations_l2 = rotations_l2 / ntest
    reflections_l2 = reflections_l2 / ntest
(
    test_time_l2
) = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None
if args.super:
    if args.super_path and full_data is None:
        indent = 1
        try:
            with h5py.File(args.super_path, "r") as f:
                data = np.array(f["u"])
            data = np.transpose(data, axes=range(len(data.shape) - 1, -1, -1))
        except:
            data = scipy.io.loadmat(os.path.expandvars(args.super_path))["u"].astype(
                np.float32
            )
        if args.nsuper:
            data = data[: args.nsuper]
        assert data.shape[1] == S_super, "wrong super space"
        assert data.shape[2] == S_super, "wrong super space"
        test_a = data[..., 3 : T_in * 4 : 4]
        test_space_u = data[..., T_in * 4 : (T + T_in) * 4 : 4]
        test_time_u = data[..., T_in * 4 : (T + T_in) * 4]
        assert test_time_u.shape[-1] == T_super, "wrong super time"
        test_space = paddle.from_numpy(
            np.concatenate([test_a, test_space_u], axis=-1)
        ).unsqueeze(-1)
        test_time = paddle.from_numpy(
            np.concatenate([test_a, test_time_u], axis=-1)
        ).unsqueeze(-1)
    elif full_data is not None:
        if args.nsuper:
            full_data = full_data[: args.nsuper]
        if rdb:
            test_space = paddle.from_numpy(full_data[..., ::4, :])
        test_time = np.concatenate(
            [full_data[..., ::4, :][..., :1, :], full_data[..., 4:, :]], axis=-2
        )
        test_time = paddle.from_numpy(test_time)
    else:
        raise ValueError("Missing super data")
    test_int_space = test_space.clone()
    test_int_time = test_time.clone()
    batch_size = 1
    test_space = pde_data(
        test_space, train=False, strategy=args.strategy, T_in=T_in, T_out=T
    )
    test_int_space = pde_data(
        test_int_space, train=False, strategy=args.strategy, T_in=T_in, T_out=T
    )
    test_time = pde_data(
        test_time, train=False, strategy=args.strategy, T_in=T_in, T_out=T_super
    )
    test_int_time = pde_data(
        test_int_time, train=False, strategy=args.strategy, T_in=T_in, T_out=T_super
    )
    space_loader = paddle.io.DataLoader(
        dataset=test_space, batch_size=batch_size, shuffle=False
    )
    space_int_loader = paddle.io.DataLoader(
        dataset=test_int_space, batch_size=batch_size, shuffle=False
    )
    time_loader = paddle.io.DataLoader(
        dataset=test_time, batch_size=batch_size, shuffle=False
    )
    time_int_loader = paddle.io.DataLoader(
        dataset=test_int_time, batch_size=batch_size, shuffle=False
    )
    ntest_super = len(space_loader)
    test_time_l2 = 0
    test_int_time_l2 = 0
    test_space_l2 = 0
    test_int_space_l2 = 0
    space_permute_inds = [0, 3, 1, 2]
    space_unpermute_inds = [0, 2, 3, 1]
    space_int_size = [*((S_super,) * d)]
    time_permute_inds = [0, 4, 1, 2, 3]
    time_unpermute_inds = [0, 2, 3, 4, 1]
    time_int_size = [*((S_super,) * d), T_super]
    with paddle.no_grad():
        for xx, yy in space_loader:
            xx, yy = move_batch(xx, yy)
            pred = get_eval_pred(
                model=model, x=xx, strategy=args.strategy, T=T, times=[]
            ).view(len(xx), *((S_super,) * d), T, num_channels)
            test_space_l2 += lploss(
                pred.reshape(len(pred), -1, num_channels),
                yy.reshape(len(yy), -1, num_channels),
            ).item()
        for xx, yy in space_int_loader:
            if rdb:
                xx = to_runtime(
                    sampler(xx.view(1, S_super, S_super, -1).permute(0, 3, 1, 2))
                    .permute(0, 2, 3, 1)
                    .view((1, *x_shape[1:]))
                )
            else:
                xx = to_runtime(xx[:, ::4, ::4])
            yy = to_runtime(yy)
            pred = get_eval_pred(
                model=model, x=xx, strategy=args.strategy, T=T, times=[]
            ).view(len(xx), *((S,) * d), T)
            pred = paddle.nn.functional.interpolate(
                pred.permute(space_permute_inds), size=space_int_size, mode="bilinear"
            ).permute(space_unpermute_inds)
            test_int_space_l2 += lploss(
                pred.reshape(len(pred), -1, num_channels),
                yy.reshape(len(yy), -1, num_channels),
            ).item()
        for xx, yy in time_loader:
            xx, yy = move_batch(xx, yy)
            pred = get_eval_pred(
                model=model, x=xx, strategy=args.strategy, T=T_super, times=[]
            ).view(len(xx), *((S_super,) * d), T_super, num_channels)
            test_time_l2 += lploss(
                pred.reshape(len(pred), -1, num_channels),
                yy.reshape(len(yy), -1, num_channels),
            ).item()
        x_new_shape = x_shape
        if threeD:
            x_new_shape[len(spatial_dims) + 1] = T_super
        x_new_shape[0] = 1
        for xx, yy in time_int_loader:
            if rdb:
                xx = to_runtime(
                    sampler(xx.view(1, S_super, S_super, -1).permute(0, 3, 1, 2))
                    .permute(0, 2, 3, 1)
                    .view(x_new_shape)
                )
            else:
                xx = to_runtime(xx[:, ::4, ::4])
            if threeD:
                xx = xx[:, :, :, ::4]
            yy = to_runtime(yy)
            pred = get_eval_pred(
                model=model, x=xx, strategy=args.strategy, T=T, times=[]
            ).view(len(xx), *((S,) * d), T, num_channels)
            pred = paddle.nn.functional.interpolate(
                pred.permute(time_permute_inds), size=time_int_size, mode="trilinear"
            ).permute(time_unpermute_inds)
            test_int_time_l2 += lploss(
                pred.reshape(len(pred), -1, num_channels),
                yy.reshape(len(yy), -1, num_channels),
            ).item()
    test_space_l2 = test_space_l2 / ntest_super
    test_int_space_l2 = test_int_space_l2 / ntest_super
    test_time_l2 = test_time_l2 / ntest_super
    test_int_time_l2 = test_int_time_l2 / ntest_super
print(
    f"""{args.model_type} done training; 
Test: {test_l2 / ntest}, Rotations: {rotations_l2}, Reflections: {reflections_l2}, Super Space Test: {test_space_l2}, Super Time Test: {test_time_l2}"""
)
summary = f"""Args: {str(args)}
Parameters: {complex_ct}
Train time: {train_time}
Mean epoch time: {train_times}
Mean inference time: {eval_times}
Num inferences: {num_eval}
Train: {train_l2 / ntrain}
Valid: {valid_l2 / nvalid}
Test: {test_l2 / ntest}
Rotation Test: {test_rt_l2 / ntest}
Reflection Test: {test_rf_l2 / ntest}
Super Space Test: {test_space_l2}
Super Space Interpolation Test: {test_int_space_l2}
Super S: {S_super}
Super Time Test: {test_time_l2}
Super Time Interpolation Test: {test_int_time_l2}
Super T: {T_super}
Best Valid: {best_valid / nvalid}
Best epoch: {best_epoch + 1}
Test Rotation Equivariance loss: {rotations_l2}
Test Reflection Equivariance loss: {reflections_l2}
Epochs trained: {ep}"""
if swe:
    summary += f"\nVorticity Test: {test_vort_l2 / ntest}\nPressure Test: {test_pres_l2 / ntest}"
txt = "results"
if args.txt_suffix:
    txt += f"_{args.txt_suffix}"
txt += ".txt"
with open(os.path.join(root, txt), "w") as f:
    f.write(summary)
