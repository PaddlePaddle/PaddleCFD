import json
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view
from paddle import summary
from sklearn.decomposition import TruncatedSVD
from utils.architecture import Unet
from utils.diffusion import ElucidatedDiffusion


matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def setup_seed(seed):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    paddle.set_flags({"FLAGS_cudnn_deterministic": True})
    paddle.set_flags({"FLAGS_benchmark": False})


def check_choose_GPU():
    print("Using device:", paddle.get_device())
    print("Number of GPUs available:", paddle.device.cuda.device_count())

    if paddle.device.cuda.device_count() >= 1:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "All visible")
        device_str = paddle.device.get_device()
        if ":" in device_str:
            gpu_id = device_str.split(":")[-1]
            gpu_name = paddle.device.cuda.get_device_properties(f"gpu:{gpu_id}").name
            print(f"CUDA_visible_devices: {visible_devices}")
            print(f"Using mapped GPU ID: {gpu_id}, Name: {gpu_name}")
        else:
            print(f"Current device is {device_str}, not GPU")
    else:
        print("CUDA is not available.")


def compute_power(true, pred):
    true = true.to("cpu")
    pred = pred.to("cpu")
    BS, nt, nx, ny = true.shape
    fourier_true = paddle.fft.fftn(x=true, axes=(-2, -1))
    fourier_pred = paddle.fft.fftn(x=pred, axes=(-2, -1))
    amplitudes_true = paddle.abs(x=fourier_true) ** 2
    amplitudes_pred = paddle.abs(x=fourier_pred) ** 2
    kfreq_y = paddle.fft.fftfreq(n=ny) * ny
    kfreq_x = paddle.fft.fftfreq(n=nx) * nx
    kfreq2D_x, kfreq2D_y = paddle.meshgrid(kfreq_x, kfreq_y)
    knrm = paddle.sqrt(x=kfreq2D_x**2 + kfreq2D_y**2)
    kbins = paddle.arange(start=0.5, end=max(nx, ny) // 2 + 1, step=1.0)
    knrm_flat = knrm.flatten()
    bin_indices = paddle.bucketize(x=knrm_flat, sorted_sequence=kbins)
    amplitudes_true_flat = amplitudes_true.view([BS, nt, nx * ny])
    amplitudes_pred_flat = amplitudes_pred.view([BS, nt, nx * ny])
    Abins_true = paddle.zeros([BS, nt, len(kbins) - 1])
    Abins_pred = paddle.zeros([BS, nt, len(kbins) - 1])
    for bin_idx in range(1, len(kbins)):
        mask = (bin_indices == bin_idx).unsqueeze(axis=0).unsqueeze(axis=0).astype("float32")
        Abins_true[:, :, bin_idx - 1] = (amplitudes_true_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
        Abins_pred[:, :, bin_idx - 1] = (amplitudes_pred_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
    scaling_factor = paddle.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins_true *= scaling_factor
    Abins_pred *= scaling_factor
    return Abins_true, Abins_pred


def compute_power_inp(true, pred, inp):
    BS, nt, nx, ny = true.shape
    true = paddle.to_tensor(data=true, dtype=data_type)
    pred = paddle.to_tensor(data=pred, dtype=data_type)
    inp = paddle.to_tensor(data=inp, dtype=data_type)
    fourier_true = paddle.fft.fftn(x=true, axes=(-2, -1))
    fourier_pred = paddle.fft.fftn(x=pred, axes=(-2, -1))
    fourier_inp = paddle.fft.fftn(x=inp, axes=(-2, -1))
    amplitudes_true = paddle.abs(x=fourier_true) ** 2
    amplitudes_pred = paddle.abs(x=fourier_pred) ** 2
    amplitudes_inp = paddle.abs(x=fourier_inp) ** 2
    kfreq_y = paddle.fft.fftfreq(n=ny) * ny
    kfreq_x = paddle.fft.fftfreq(n=nx) * nx
    kfreq2D_x, kfreq2D_y = paddle.meshgrid(kfreq_x, kfreq_y)
    knrm = paddle.sqrt(x=kfreq2D_x**2 + kfreq2D_y**2).to(true.place)
    kbins = paddle.arange(start=0.5, end=nx // 2 + 1, step=1.0)
    knrm_flat = knrm.flatten()
    bin_indices = paddle.bucketize(x=knrm_flat, sorted_sequence=kbins)
    amplitudes_true_flat = amplitudes_true.view([BS, nt, nx * ny])
    amplitudes_pred_flat = amplitudes_pred.view([BS, nt, nx * ny])
    amplitudes_inp_flat = amplitudes_inp.view([BS, nt, nx * ny])
    Abins_true = paddle.zeros([BS, nt, len(kbins) - 1])
    Abins_pred = paddle.zeros([BS, nt, len(kbins) - 1])
    Abins_inp = paddle.zeros([BS, nt, len(kbins) - 1])
    for bin_idx in range(1, len(kbins)):
        mask = (bin_indices == bin_idx).unsqueeze(axis=0).unsqueeze(axis=0).astype("float32")
        Abins_true[:, :, bin_idx - 1] = (amplitudes_true_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
        Abins_pred[:, :, bin_idx - 1] = (amplitudes_pred_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
        Abins_inp[:, :, bin_idx - 1] = (amplitudes_inp_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
    scaling_factor = paddle.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins_true *= scaling_factor
    Abins_pred *= scaling_factor
    Abins_inp *= scaling_factor
    Abins_true = paddle.log(x=Abins_true)
    Abins_pred = paddle.log(x=Abins_pred)
    Abins_inp = paddle.log(x=Abins_inp)
    return (
        Abins_true.numpy(),
        Abins_pred.numpy(),
        Abins_inp.numpy(),
    )


def plot_power_spectrum(power_true, power_pred, epoch, err):
    f = 2
    power_true = power_true[4:10, 0]
    power_pred = power_pred[4:10, 0]
    fig, axes = plt.subplots(1, 6, figsize=(8 * f, 1 * f))
    t_ls = np.arange(4, 10)
    time_ls = t_ls
    # sample_id = 0   # unused
    for i in range(6):
        x = np.arange(0, power_true.shape[-1], 1)
        axes[i].loglog(x, power_true[i], label="true")
        axes[i].loglog(x, power_pred[i], label="pred")
        axes[i].set_title(f"t: {time_ls[i]}")
        axes[i].set_xlabel("$k$")
        if i == 5:
            axes[i].legend()
        if i == 0:
            axes[i].set_ylabel("$P(k)$")
    fig.suptitle(f"Epoch: {epoch}, MSE: {err:.2e}", fontsize=22, y=1.2)
    os.makedirs("power_spectrum", exist_ok=True)
    plt.savefig(f"power_spectrum/{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close()


def error_metric(pred, true, epoch, Par, is_plot=False):
    power_true, power_pred = compute_power(true, pred)
    err = paddle.mean(x=(paddle.log(x=power_true) - paddle.log(x=power_pred)) ** 2)
    if is_plot:
        plot_power_spectrum(power_true.numpy(), power_pred.numpy(), epoch, err)
    return err


def field_get_err(true, pred):
    return np.mean(np.mean((true - pred) ** 2, axis=(2, 3)) / np.mean(true**2, axis=(2, 3)))


def spec_get_err(true, pred):
    return np.mean(np.mean((true - pred) ** 2, axis=(1, 2)) / np.mean(true**2, axis=(1, 2)))


def mse(a, b):
    return np.mean((a - b) ** 2)


def rel_l2(T, P):
    T = paddle.to_tensor(data=T, dtype=data_type)
    P = paddle.to_tensor(data=P, dtype=data_type)
    return paddle.linalg.norm(x=T - P, p=2) / paddle.linalg.norm(x=T, p=2).cpu().numpy()


class MyDataset(paddle.io.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)
        return x_sample, y_sample


def preprocess(x, y, Par):
    x = (
        sliding_window_view(x[:, Par["lb"] - 1 :, :, :], window_shape=Par["lf"], axis=1)
        .transpose(0, 1, 4, 2, 3)
        .reshape([-1, Par["lf"], Par["nx"], Par["ny"]])
    )
    y = (
        sliding_window_view(y[:, Par["lb"] - 1 :, :, :], window_shape=Par["lf"], axis=1)
        .transpose(0, 1, 4, 2, 3)
        .reshape([-1, Par["lf"], Par["nx"], Par["ny"]])
    )
    return x, y


def do_svd(data):
    TRAJ = data.reshape([data.shape[0], -1])
    ld = TRAJ.shape[0]
    n_components = ld - 1
    t_svd = TruncatedSVD(n_components=n_components)
    s_train = t_svd.singular_values_
    v_train = t_svd.components_
    return s_train, v_train


if __name__ == "__main__":
    seed_value = 23
    data_type = "float32"
    res = 128
    path_model = "models/best_model.pdparams"
    check_choose_GPU()

    debug = False

    scaler = paddle.amp.GradScaler(incr_every_n_steps=2000, init_loss_scaling=65536.0)

    begin_time = time.time()

    train_pred_dir = "../TRAIN_PRED.npy"
    train_true_dir = "../TRAIN_TRUE.npy"
    x_train = np.load(train_pred_dir)
    y_train = np.load(train_true_dir)

    val_pred_dir = "../VAL_PRED.npy"
    val_true_dir = "../VAL_TRUE.npy"
    x_val = np.load(val_pred_dir)
    y_val = np.load(val_true_dir)

    test_pred_dir = "../TEST_PRED.npy"
    test_true_dir = "../TEST_TRUE.npy"
    x_test = np.load(test_pred_dir)
    y_test = np.load(test_true_dir)

    if debug:
        x_train = x_train[0:20]
        y_train = y_train[0:20]
        x_val = x_val[0:4]
        y_val = y_val[0:4]
        x_test = x_test[0:5]
        y_test = y_test[0:5]

    inp_min = np.min(x_train)
    inp_max = np.max(x_train)
    out_min = np.min(y_train)
    out_max = np.max(y_train)

    Par = {
        "inp_shift": paddle.to_tensor(data=inp_min, dtype=data_type),
        "inp_scale": paddle.to_tensor(data=inp_max - inp_min, dtype=data_type),
        "out_shift": paddle.to_tensor(data=out_min, dtype=data_type),
        "out_scale": paddle.to_tensor(data=out_max - out_min, dtype=data_type),
        "nx": x_train.shape[2],
        "ny": x_train.shape[3],
        "nf": 1,
        "lb": 1,
        "lf": 1,
        "num_epochs": 1000,
    }

    # Normalizing the data to [0,1]
    shift = Par["inp_shift"].numpy()
    scale = Par["inp_scale"].numpy()
    x_train = (x_train - shift) / scale
    x_val = (x_val - shift) / scale
    x_test = (x_test - shift) / scale

    shift = Par["out_shift"].numpy()
    scale = Par["out_scale"].numpy()
    y_train = (y_train - shift) / scale
    y_val = (y_val - shift) / scale
    y_test = (y_test - shift) / scale

    Par["sigma_data"] = np.std(y_train)

    begin_time = time.time()
    x_train, y_train = preprocess(x_train, y_train, Par)
    x_val, y_val = preprocess(x_val, y_val, Par)
    x_test, y_test = preprocess(x_test, y_test, Par)
    print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

    Par.update({"channels": x_train.shape[1], "self_condition": True})

    def tensor_to_serializable(obj):
        if isinstance(obj, paddle.Tensor):
            return float(obj.numpy().item()) if obj.ndim == 0 else obj.numpy().tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        return obj

    with open("Par.json", "w") as f:
        json.dump({k: tensor_to_serializable(v) for k, v in Par.items()}, f)

    x_train_tensor = paddle.to_tensor(data=x_train, dtype="float32")
    y_train_tensor = paddle.to_tensor(data=y_train, dtype="float32")
    x_val_tensor = paddle.to_tensor(data=x_val, dtype="float32")
    y_val_tensor = paddle.to_tensor(data=y_val, dtype="float32")
    x_test_tensor = paddle.to_tensor(data=x_test, dtype="float32")
    y_test_tensor = paddle.to_tensor(data=y_test, dtype="float32")

    train_dataset = MyDataset(x_train_tensor, y_train_tensor)
    val_dataset = MyDataset(x_val_tensor, y_val_tensor)
    test_dataset = MyDataset(x_test_tensor, y_test_tensor)

    # Define data loaders
    """
    Cannot be set to 100, because of error:
    RuntimeError: (PreconditionNotMet) The element size of transformed_output should be <= INT_MAX(2147483647),
    but got 2516582400 (=100*384*128*512) [Hint: Expected transformed_output.numel() <= largest,
    but received transformed_output.numel():2516582400 > largest:2147483647.]
    (at ../paddle/phi/kernels/gpudnn/conv_kernel.cu:500)
    """
    train_batch_size = 20  # 16
    val_batch_size = 20
    test_batch_size = 20
    print(f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}")
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
    val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=val_batch_size)
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=test_batch_size)

    # Define Network Architecture
    net = Unet(
        dim=16,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
        self_condition=Par["self_condition"],
        flash_attn=True,
    ).astype("float32")
    summary(net, input_size=[(1,) + x_train.shape[1:], (1,)])

    model = ElucidatedDiffusion(
        net,
        channels=Par["channels"],
        image_size_h=Par["nx"],
        image_size_w=Par["ny"],
        sigma_data=Par["sigma_data"],
    )
    model.set_state_dict(state_dict=paddle.load(path=str(path_model)))

    epoch = 0
    l_fidel_ls = []
    y_pred_ls = []
    y_true_ls = []

    model.eval()
    test_loss = 0.0
    with paddle.no_grad():
        for l_fidel, h_fidel in test_loader:
            l_fidel_ls.append(l_fidel.clone())
            y_true_ls.append(h_fidel.clone())
            with paddle.amp.auto_cast(enable=False):
                pred = model.sample(l_fidel, num_sample_steps=32)
                y_pred_ls.append(pred.clone())
                loss = error_metric(pred, h_fidel, epoch, Par, is_plot=True)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4e}\n")

    y_true = paddle.concat(x=y_true_ls, axis=0).numpy().reshape([-1, 10, Par["nx"], Par["ny"]])
    y_pred = paddle.concat(x=y_pred_ls, axis=0).numpy().reshape([-1, 10, Par["nx"], Par["ny"]])
    y_no = paddle.concat(x=l_fidel_ls, axis=0).numpy().reshape([-1, 10, Par["nx"], Par["ny"]])

    print("Renormalizing the output")
    y_true = y_true * Par["out_scale"].numpy() + Par["out_shift"].numpy()
    y_pred = y_pred * Par["out_scale"].numpy() + Par["out_shift"].numpy()
    y_no = y_no * Par["inp_scale"].numpy() + Par["inp_shift"].numpy()

    spec_true, spec_pred, spec_no = compute_power_inp(y_true, y_pred, y_no)
    field_err_pred = field_get_err(y_true, y_pred)
    field_err_no = field_get_err(y_true, y_no)
    print(f"Field error: nodm = {field_err_pred:.4e}, no = {field_err_no:.4e}")
    spec_err_pred = spec_get_err(spec_true, spec_pred)
    spec_err_no = spec_get_err(spec_true, spec_no)
    print(f"Spec error : nodm = {spec_err_pred:.4e}, no = {spec_err_no:.4e}\n")

    t_ls = np.linspace(0, 10, 11)[1:]
    mask = np.load("../../data/mask.npy")

    Lc = 1
    Nx = 320
    Ny = 128
    Lx = 320 / 512 * 6.5 * Lc
    Ly = 2.5 * Lc
    delta_x = Lx / Nx
    delta_y = Ly / Ny
    k_x_nq = np.pi / delta_x
    k_y_nq = np.pi / delta_y
    k_nq = min(k_x_nq, k_y_nq)

    for p in range(0, 100, 1):
        sample_id = p
        print(f"Sample id: {sample_id}")
        skip_t = 4
        t_ls = np.arange(11)[1:][1:] * 5
        true = y_true[sample_id, 1:][::skip_t][1:, :Ny, :Nx]
        pred1 = y_no[sample_id, 1:][::skip_t][1:, :Ny, :Nx]
        pred2 = y_pred[sample_id, 1:][::skip_t][1:, :Ny, :Nx]
        time_ls = t_ls[::skip_t][1:]
        print(f"Shape of true: {true.shape}")
        print(f"Time step to plot: {time_ls}")

        CMAP = "turbo"
        fig, axes = plt.subplots(
            4,
            len(time_ls),
            figsize=(14, 11),
            gridspec_kw={"height_ratios": [1, 1, 1, 1.5]},
        )
        cbar_ax = fig.add_axes([0.9, 0.37, 0.02, 0.57])

        vmin = min(true.min(), pred1.min(), pred2.min())
        vmax = max(true.max(), pred1.max(), pred2.max())

        airfoil_mask = np.expand_dims(mask, axis=0)
        airfoil_mask = np.tile(airfoil_mask, (true.shape[0], 1, 1))

        true_temp = np.copy(true)
        pred1_temp = np.copy(pred1)
        pred2_temp = np.copy(pred2)

        for i in range(len(time_ls)):

            im = axes[0, i].imshow(true_temp[i], origin="lower", vmin=vmin, vmax=vmax, cmap=CMAP)
            delta_t = f"{time_ls[i]}"
            tau_component = f"{time_ls[i] / 64:.1f}"
            title = f"$\\Delta t$: {delta_t}$\\tau$ (={tau_component}$T_{{st}}$)"
            axes[0, i].set_title(title, fontsize=24)
            axes[0, i].axis("off")
            im = axes[1, i].imshow(pred1_temp[i], origin="lower", vmin=vmin, vmax=vmax, cmap=CMAP)
            axes[1, i].axis("off")
            im = axes[2, i].imshow(pred2_temp[i], origin="lower", vmin=vmin, vmax=vmax, cmap=CMAP)
            axes[2, i].axis("off")

            image = true[i]
            ny, nx = image.shape
            # Compute the Fourier transform and get the amplitude squared
            fourier_image = np.fft.fftn(image)
            fourier_amplitudes = np.abs(fourier_image) ** 2
            # Create the k-frequency grid
            kfreq_y = np.fft.fftfreq(ny) * ny
            kfreq_x = np.fft.fftfreq(nx) * nx
            kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
            # Flatten the arrays to use in binning
            knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
            knrm = knrm.flatten()
            fourier_amplitudes = fourier_amplitudes.flatten()
            # Define the bins for the wavenumber
            kbins = np.arange(0.5, min(nx, ny) // 2 + 1, 1.0)
            kvals = 0.5 * (kbins[1:] + kbins[:-1])
            kvals = kvals / (0.5 * min(nx, ny)) * k_nq
            # Bin the data
            Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
            # Scale the binned amplitudes
            Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
            # Plotting
            axes[3, i].loglog(kvals, Abins, c="black", label="Ground Truth", linewidth=3)

            image = pred1[i]
            ny, nx = image.shape
            # Compute the Fourier transform and get the amplitude squared
            fourier_image = np.fft.fftn(image)
            fourier_amplitudes = np.abs(fourier_image) ** 2
            # Create the k-frequency grid
            kfreq_y = np.fft.fftfreq(ny) * ny
            kfreq_x = np.fft.fftfreq(nx) * nx
            kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
            # Flatten the arrays to use in binning
            knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
            knrm = knrm.flatten()
            fourier_amplitudes = fourier_amplitudes.flatten()
            # Define the bins for the wavenumber
            kbins = np.arange(0.5, min(nx, ny) // 2 + 1, 1.0)
            kvals = 0.5 * (kbins[1:] + kbins[:-1])
            kvals = kvals / (0.5 * min(nx, ny)) * k_nq
            # Bin the data
            Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
            # Scale the binned amplitudes
            Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
            # Plotting
            axes[3, i].loglog(kvals, Abins, label="NO", c="blue", linewidth=2)

            image = pred2[i]
            ny, nx = image.shape
            fourier_image = np.fft.fftn(image)
            fourier_amplitudes = np.abs(fourier_image) ** 2
            kfreq_y = np.fft.fftfreq(ny) * ny
            kfreq_x = np.fft.fftfreq(nx) * nx
            kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
            knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
            knrm = knrm.flatten()
            fourier_amplitudes = fourier_amplitudes.flatten()
            kbins = np.arange(0.5, min(nx, ny) // 2 + 1, 1.0)
            kvals = 0.5 * (kbins[1:] + kbins[:-1])
            kvals = kvals / (0.5 * min(nx, ny)) * k_nq
            Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
            Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
            axes[3, i].loglog(kvals, Abins, label="NO+Diffusion", c="red", linewidth=2)

            axes[3, i].set_ylim(9 * 10**2, 2 * 10**8)
            for ax in axes[3, :]:
                ax.tick_params(axis="both", which="major", labelsize=17)
            if i != 0:
                axes[3, i].get_yaxis().set_visible(False)

            # Adding the -5/3 slope line
            k_ref = np.linspace(1, np.max(kbins), 100)
            energy_ref = k_ref ** (-5 / 3)
            energy_ref *= max(Abins) / max(energy_ref)

            axes[3, i].set_xlabel("$k$", fontsize=20)
            if i == 1:
                leg = axes[3, i].legend()
                handles, labels = axes[3, i].get_legend_handles_labels()
                leg.remove()
                custom_x = 0.81
                custom_y = 0.1
                fig.legend(
                    handles,
                    labels,
                    loc="lower left",
                    bbox_to_anchor=(custom_x, custom_y),
                    fontsize=19,
                )
            axes[3, i].set_aspect(1 / 3)

        # Add row labels
        kwargs = {"va": "center", "ha": "center", "rotation": "vertical"}
        fig.text(0.04, 0.87, "Ground truth", fontsize=18, **kwargs)
        fig.text(0.04, 0.67, "Neural Operator", fontsize=18, **kwargs)
        fig.text(0.02, 0.46, "Neural Operator\n+\nDiffusion Model", fontsize=18, **kwargs)
        fig.text(0.06, 0.20, r"Energy $P(k)$", fontsize=20, **kwargs)
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=17)
        plt.savefig("u-contour-3models.png", dpi=600, bbox_inches="tight")
        plt.close()

    t_idx = 1
    sample_id = 3
    print("TRUE")
    s_true, v_true = do_svd(y_true[sample_id, :, :Ny, :Nx])
    print("\nNO")
    s_no, v_no = do_svd(y_no[sample_id, :, :Ny, :Nx])
    print("\nPRED")
    s_pred, v_pred = do_svd(y_pred[sample_id, :, :Ny, :Nx])

    fig = plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(s_true) + 1), s_true / np.sum(s_true), c="black", label="Ground Truth")
    plt.plot(range(1, len(s_no) + 1), s_no / np.sum(s_no), c="blue", label="NO")
    plt.plot(range(1, len(s_pred) + 1), s_pred / np.sum(s_pred), c="red", label="NO+Diffusion")

    plt.ylabel("Energy " + "$\\frac{\\lambda_i}{\\sum_j \\lambda_j}$", fontsize=24)
    plt.xlabel("Mode i", fontsize=24)
    plt.legend(fontsize=18)
    plt.yscale("log")
    plt.ylim(0.0003153880729691828, 1.3612854157338017)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("mode-energy.png", dpi=600, bbox_inches="tight")
    plt.close()

    mode_idx = [0, 2, 4, 6, 8]
    V_TRUE = v_true.reshape([-1, Ny, Nx])[mode_idx]
    V_NO = v_no.reshape([-1, Ny, Nx])[mode_idx]
    V_PRED = v_pred.reshape([-1, Ny, Nx])[mode_idx]

    n = V_TRUE.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(12, 8))
    CMAP = "gray"
    for i in range(n):
        # Plot V_TRUE
        vmin_row = min(np.min(V_TRUE[i]), np.min(V_NO[i]), np.min(V_PRED[i]))
        vmax_row = max(np.max(V_TRUE[i]), np.max(V_NO[i]), np.max(V_PRED[i]))
        im = axes[i, 0].imshow(V_TRUE[i], origin="lower", cmap=CMAP, vmin=vmin_row, vmax=vmax_row)
        axes[i, 0].set_ylabel(f"Mode {mode_idx[i] + 1}")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Plot V_NO
        im = axes[i, 1].imshow(V_NO[i], origin="lower", cmap=CMAP, vmin=vmin_row, vmax=vmax_row)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Plot V_PRED
        im = axes[i, 2].imshow(V_PRED[i], origin="lower", cmap=CMAP, vmin=vmin_row, vmax=vmax_row)
        cbar = plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04, aspect=10)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        if i == 0:
            axes[i, 0].set_title("Ground Truth")
            axes[i, 1].set_title("Neural Operator")
            axes[i, 2].set_title("Neural Operator\n+\nDiffusion Model")

    plt.tight_layout()
    plt.savefig("POD-diffMode.png", dpi=600, bbox_inches="tight")
    plt.close()
