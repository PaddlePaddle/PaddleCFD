import json
import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import paddle
from numpy.lib.stride_tricks import sliding_window_view
from paddle import summary
from tqdm import tqdm

from ppcfd.models.ppdiffusion.utils.architecture import Unet
from ppcfd.models.ppdiffusion.utils.diffusion import ElucidatedDiffusion


def setup_seed(seed):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    paddle.set_flags({"FLAGS_cudnn_deterministic": True})
    paddle.set_flags({"FLAGS_benchmark": False})


def init_all(seed, name, dtype):
    setup_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    paddle.set_default_dtype(d=dtype)
    if not os.path.exists(name):
        os.makedirs(name)
    log_level = logging.INFO
    log_name = os.path.join(name, time.strftime("%Y-%m-%d-%H-%M-%S") + ".log")
    logger = logging.getLogger("")
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_name, encoding="utf8")
    file_handler.setLevel(level=log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Project name: {name}")
    logger.info(f"Random seed value: {seed}, data type : {dtype}\n")
    return logger


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


def compute_power(true, pred, inp):
    BS, nt, nx, ny = true.shape

    # Compute the Fourier transforms and amplitude squared for both true and pred
    fourier_true = paddle.fft.fftn(x=true, axes=(-2, -1))
    fourier_pred = paddle.fft.fftn(x=pred, axes=(-2, -1))
    fourier_inp = paddle.fft.fftn(x=inp, axes=(-2, -1))

    # Get the squared amplitudes
    amplitudes_true = paddle.abs(x=fourier_true) ** 2
    amplitudes_pred = paddle.abs(x=fourier_pred) ** 2
    amplitudes_inp = paddle.abs(x=fourier_inp) ** 2

    # Create the k-frequency grids
    kfreq_y = paddle.fft.fftfreq(n=ny) * ny
    kfreq_x = paddle.fft.fftfreq(n=nx) * nx
    kfreq2D_x, kfreq2D_y = paddle.meshgrid(kfreq_x, kfreq_y)

    # Compute the wavenumber grid
    knrm = paddle.sqrt(x=kfreq2D_x**2 + kfreq2D_y**2).to(true.place)

    # Define the bins for the wavenumber
    kbins = paddle.arange(start=0.5, end=nx // 2 + 1, step=1.0)

    # Digitize knrm to bin indices
    knrm_flat = knrm.flatten()
    bin_indices = paddle.bucketize(x=knrm_flat, sorted_sequence=kbins)

    # Reshape and flatten the amplitudes
    amplitudes_true_flat = amplitudes_true.view([BS, nt, nx * ny])
    amplitudes_pred_flat = amplitudes_pred.view([BS, nt, nx * ny])
    amplitudes_inp_flat = amplitudes_inp.view([BS, nt, nx * ny])

    # Initialize Abins
    Abins_true = paddle.zeros(shape=(BS, nt, len(kbins) - 1))
    Abins_pred = paddle.zeros(shape=(BS, nt, len(kbins) - 1))
    Abins_inp = paddle.zeros(shape=(BS, nt, len(kbins) - 1))

    # Vectorized binning: sum up the values in each bin
    for bin_idx in range(1, len(kbins)):
        mask = (
            (bin_indices == bin_idx).unsqueeze(axis=0).unsqueeze(axis=0).astype("float32")
        )  # Create a mask for each bin
        Abins_true[:, :, bin_idx - 1] = (amplitudes_true_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
        Abins_pred[:, :, bin_idx - 1] = (amplitudes_pred_flat * mask).sum(axis=-1) / mask.sum(axis=-1)
        Abins_inp[:, :, bin_idx - 1] = (amplitudes_inp_flat * mask).sum(axis=-1) / mask.sum(axis=-1)

    # Scale the binned amplitudes
    scaling_factor = paddle.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins_true *= scaling_factor
    Abins_pred *= scaling_factor
    Abins_inp *= scaling_factor

    return Abins_true, Abins_pred, Abins_inp


def plot_power_spectrum(power_inp, power_true, power_pred, inp, true, pred, epoch, err):
    CMAP = "turbo"
    f = 2
    fig, axes = plt.subplots(1, 4, figsize=(4 * f, 1 * f))

    sample_id = 0
    for i in range(1):
        x = np.arange(0, int(true.shape[-2] // 2), 1)
        axes[0].loglog(x, power_true[sample_id, 0], label="true", c="black")
        axes[0].loglog(x, power_inp[sample_id, 0], label="NO", c="blue")
        axes[0].loglog(x, power_pred[sample_id, 0], label="NO+DM", c="red")
        axes[0].set_xlabel("$k$")
        if i == 0:
            axes[0].legend()
        if i == 0:
            axes[0].set_ylabel("$P(k)$")

    inp_sample = inp[sample_id, 0]
    true_sample = true[sample_id, 0]
    pred_sample = pred[sample_id, 0]
    vmin, vmax = true_sample.min(), true_sample.max()
    im1 = axes[1].imshow(true_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[1].set_title("True")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # im = axes[2].imshow(inp_sample, vmin=vmin, vmax=vmax, cmap=CMAP)    # im unused
    axes[2].imshow(inp_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[2].set_title("NO")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # im = axes[3].imshow(pred_sample, vmin=vmin, vmax=vmax, cmap=CMAP)    # im unused
    axes[3].imshow(pred_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[3].set_title("NO+DM")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im1, ax=axes[3])
    plt.tight_layout()

    fig.suptitle(f"Epoch: {epoch}, MSE: {err:.2e}", fontsize=22, y=1.2)
    plt.savefig(f"power_spectrum/{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close()


def error_metric(inp, pred, true, epoch, Par, is_plot=True):
    inp = inp * Par["inp_scale"] + Par["inp_shift"]
    inp = (inp - Par["out_shift"]) / Par["out_scale"]

    power_inp, power_true, power_pred = compute_power(inp, true, pred)
    err = paddle.mean(x=(paddle.log(x=power_true) - paddle.log(x=power_pred)) ** 2)
    f_err = paddle.linalg.norm(x=true - pred, p=2) / paddle.linalg.norm(x=true, p=2)
    ref_err = paddle.linalg.norm(x=true - inp, p=2) / paddle.linalg.norm(x=true, p=2)
    if is_plot:
        plot_power_spectrum(
            power_inp.numpy(),
            power_true.numpy(),
            power_pred.numpy(),
            inp.numpy(),
            true.numpy(),
            pred.numpy(),
            epoch,
            err,
        )
    return err, f_err, ref_err


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
        .reshape(-1, Par["lf"], Par["nx"], Par["ny"])
    )
    y = (
        sliding_window_view(y[:, Par["lb"] - 1 :, :, :], window_shape=Par["lf"], axis=1)
        .transpose(0, 1, 4, 2, 3)
        .reshape(-1, Par["lf"], Par["nx"], Par["ny"])
    )
    return x, y


if __name__ == "__main__":
    seed_value = 23
    data_type = "float32"
    save_dir = f"seed_{seed_value}"
    res = 128
    debug = False

    logger = init_all(seed_value, name=save_dir, dtype=data_type)
    scaler = paddle.amp.GradScaler(incr_every_n_steps=2000, init_loss_scaling=65536.0)
    begin_time = time.time()

    train_pred_dir = "../TRAIN_PRED.npy"
    train_true_dir = "../TRAIN_TRUE.npy"
    logger.info(f"Loading train data from: {train_pred_dir}, {train_true_dir}")
    x_train = np.load(train_pred_dir)
    y_train = np.load(train_true_dir)
    logger.info(f"Shape of train data: {x_train.shape}")

    val_pred_dir = "../VAL_PRED.npy"
    val_true_dir = "../VAL_TRUE.npy"
    logger.info(f"Loading val data from: {val_pred_dir}, {val_true_dir}")
    x_val = np.load(val_pred_dir)
    y_val = np.load(val_true_dir)
    logger.info(f"Shape of val data: {x_val.shape}")

    test_pred_dir = "../TEST_PRED.npy"
    test_true_dir = "../TEST_TRUE.npy"
    logger.info(f"Loading test data from: {test_pred_dir}, {test_true_dir}")
    x_test = np.load(test_pred_dir)
    y_test = np.load(test_true_dir)
    logger.info(f"Shape of test data: {x_test.shape}")

    if debug:
        x_train = x_train[0:20]
        y_train = y_train[0:20]
        x_val = x_val[0:4]
        y_val = y_val[0:4]
        x_test = x_test[0:4]
        y_test = y_test[0:4]
    logger.info(f"Data loading time: {time.time() - begin_time:.1f}s\n")

    inp_min = np.min(x_train)
    inp_max = np.max(x_train)
    out_min = np.min(y_train)
    out_max = np.max(y_train)
    logger.info(f"Minimum of input in train data: {inp_min}")
    logger.info(f"Maximum of input in train data: {inp_max}")
    logger.info(f"Minimum of output in train data: {out_min}")
    logger.info(f"Maximum of output in train data: {out_max}")

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
        "num_epochs": 100000,
    }
    logger.info(f"Input shift: {Par['inp_shift']}")
    logger.info(f"Input scale: {Par['inp_scale']}")
    logger.info(f"Output shift: {Par['out_shift']}")
    logger.info(f"Output scale: {Par['out_scale']}")
    logger.info(f"Dimension of flow (nx*ny): ({Par['nx']}, {Par['ny']})")
    logger.info(f"Number of features (nf): {Par['nf']}")
    logger.info(f"Number of timesteps as inputs (lb): {Par['lb']}")
    logger.info(f"Number of timesteps as outputs (lf): {Par['lf']}")
    logger.info(f"Number epochs: {Par['num_epochs']}\n")

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

    # Traj splitting
    begin_time = time.time()
    x_train, y_train = preprocess(x_train, y_train, Par)
    logger.info("Shape of train dataset")
    logger.info(f"x_idx_train: {x_train.shape}")
    logger.info(f"y_idx_train: {y_train.shape}\n")

    x_val, y_val = preprocess(x_val, y_val, Par)
    logger.info("Shape of val dataset")
    logger.info(f"x_idx_val: {x_val.shape}")
    logger.info(f"y_idx_val: {y_val.shape}\n")

    x_test, y_test = preprocess(x_test, y_test, Par)
    logger.info("Shape of test dataset")
    logger.info(f"x_idx_test: {x_test.shape}")
    logger.info(f"y_idx_test: {y_test.shape}\n")
    logger.info(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

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
    train_batch_size = 20
    val_batch_size = 20
    test_batch_size = 20
    logger.info(f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}")
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
    val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=val_batch_size)
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=test_batch_size)

    net = Unet(
        dim=16,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
        self_condition=Par["self_condition"],
        flash_attn=True,
    ).astype("float32")

    # summary
    summary(net, input_size=((1,) + tuple(x_train.shape)[1:], (1,)))

    model = ElucidatedDiffusion(
        net,
        channels=Par["channels"],
        image_size_h=Par["nx"],
        image_size_w=Par["ny"],
        sigma_data=Par["sigma_data"],
    )

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001, weight_decay=0)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=Par["num_epochs"] * len(train_loader), learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(scheduler)

    # Training loop
    num_epochs = Par["num_epochs"]
    best_val_loss = float("inf")
    best_f_loss = float("inf")
    best_model_id = 0

    os.makedirs("models", exist_ok=True)
    os.makedirs("power_spectrum", exist_ok=True)

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        begin_time = time.time()
        model.train()
        train_loss = 0.0
        train_time = time.time()
        for l_fidel, h_fidel in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            optimizer.clear_gradients(set_to_zero=False)
            with paddle.amp.auto_cast(enable=False):
                loss = model(h_fidel, l_fidel)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_time = time.time() - train_time

        # Validation
        if epoch != 1 and epoch % 10 == 0:
            val_time = time.time()
            model.eval()
            val_loss = 0.0
            field_err = 0.0
            ref_err = 0.0
            with paddle.no_grad():
                for l_fidel, h_fidel in val_loader:
                    with paddle.amp.auto_cast(enable=False):
                        pred = model.sample(l_fidel)
                        loss, f_err, r_err = error_metric(l_fidel, pred, h_fidel, epoch, Par)
                    field_err += f_err.item()
                    val_loss += loss.item()
                    ref_err += r_err.item()

            val_loss /= len(val_loader)
            field_err /= len(val_loader)
            ref_err /= len(val_loader)

            if field_err < best_f_loss or field_err < ref_err:
                best_f_loss = field_err
                best_model_id = epoch
                paddle.save(obj=model.state_dict(), path=f"models/best_model_{epoch}.pdparams")

            if epoch % 100 == 0:
                paddle.save(obj=model.state_dict(), path=f"models/model_{epoch}.pdparams")
            val_time = time.time() - val_time
            logger.info(
                f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4e}, "
                f"Val Loss (Spectrum): {val_loss:.4e}, Val Loss (field): {field_err:.4e}, "
                f"Best model: {best_model_id}, Best err: {best_f_loss:.4e}, Ref err: {ref_err:.4e}, "
                f"Learning rate: {scheduler.get_lr():.4e}, "
                f"Train time: {train_time:.2f}, Val time: {val_time:.2f}"
            )
        else:
            logger.info(
                f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4e}, "
                f"Learning rate: {scheduler.get_lr():.4e}, Train time: {train_time:.2f}"
            )

    logger.info("Training finished.")
    logger.info(f"Training Time: {time.time() - t0:.1f}s\n")

    # Testing loop
    model.eval()
    total_err = 0.0
    total_f_err = 0.0
    total_ref_err = 0.0
    with paddle.no_grad():
        for l_fidel, h_fidel in test_loader:
            with paddle.amp.auto_cast(enable=False):
                pred = model.sample(l_fidel)
                err, f_err, ref_err = error_metric(l_fidel, pred, h_fidel, epoch, Par, is_plot=False)
            total_err += err.item()
            total_f_err += f_err.item()
            total_ref_err += ref_err.item()

    n_samples = len(test_loader)
    logger.info(f"L2 Error: {total_f_err / n_samples}")
    logger.info(f"Spectral Error: {total_err / n_samples}")
    logger.info(f"Reference Error: {total_ref_err / n_samples}")
