import json
import logging
import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.stats as stats
from matcho import Unet2D
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


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


def make_plot(TRUE, PRED, epoch):
    sample_id = 0
    skip_t = 4
    t_ls = np.arange(11)[1:]
    true = TRUE[sample_id, ::skip_t, 0]
    pred1 = PRED[sample_id, ::skip_t, 0]
    time_ls = t_ls[::skip_t]
    CMAP = "turbo"
    N = min(true.shape[0], 3)

    def mse(a, b):
        return np.mean((a - b) ** 2)

    def rel_l2(T, P):
        T = paddle.to_tensor(data=T, dtype=data_type)
        P = paddle.to_tensor(data=P, dtype=data_type)
        return paddle.linalg.norm(x=T - P, p=2) / paddle.linalg.norm(x=T, p=2).cpu().numpy()

    fig, axes = plt.subplots(4, N, figsize=(30, 11))
    if N == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])

    vmin = min(true.min(), pred1.min())
    vmax = max(true.max(), pred1.max())

    for i in range(N):
        im = axes[0, i].imshow(true[i], vmin=vmin, vmax=vmax, cmap=CMAP)
        axes[0, i].set_title(f"Time: {time_ls[i]}s", fontsize=16)
        axes[0, i].axis("off")
        im = axes[1, i].imshow(pred1[i], vmin=vmin, vmax=vmax, cmap=CMAP)
        mse_val1 = rel_l2(true[i], pred1[i])
        axes[1, i].set_title(f"rel L2: {mse_val1:.2e}", fontsize=12)
        axes[1, i].axis("off")
        image = true[i]
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
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        axes[2, i].loglog(kvals, Abins, label="Simulated")
        image = pred1[i]
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
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        axes[2, i].loglog(kvals, Abins, label="MATCHO")
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(Abins) / max(energy_ref)
        if i != 0:
            axes[2, i].loglog(k_ref, energy_ref, "k--", label="k^-5/3 Reference")
        axes[2, i].set_xlabel("$k$")
        if i == 5:
            axes[2, i].legend()
        image = true[i]
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
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        axes[3, i].loglog(kvals, Abins, label="Simulated")
        image = pred1[i]
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
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        axes[3, i].loglog(kvals, Abins, label="MATCHO")
        if i == 5:
            axes[2, i].legend()
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f"images/{epoch + 1}.png")


class CustomLoss(paddle.nn.Layer):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()
        self.Par = Par

    def forward(self, y_pred, y_true):
        y_true = (y_true - self.Par["out_shift"]) / self.Par["out_scale"]
        y_pred = (y_pred - self.Par["out_shift"]) / self.Par["out_scale"]
        loss = paddle.linalg.norm(x=y_true - y_pred, p=2) / paddle.linalg.norm(x=y_true, p=2)
        return loss


class YourDataset_train(paddle.io.Dataset):
    def __init__(self, x, t, y, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)
        return x_sample, t_sample, y_sample


class YourDataset(paddle.io.Dataset):
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


def preprocess_train(traj, Par):
    nt = traj.shape[1]
    temp = nt - Par["lb"] - Par["lf"] + 1
    x_idx = np.arange(temp).reshape(-1, 1)
    x_idx = np.tile(x_idx, (1, Par["lf"])).reshape(-1, 1)
    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx + i)
    x_idx = np.concatenate(x_idx_ls, axis=1)
    t_idx = np.arange(Par["lf"]).reshape(1, -1)
    t_idx = np.tile(t_idx, (temp, 1)).reshape(-1)
    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=Par["lf"]).reshape(-1)
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def preprocess(traj, Par):
    nt = traj.shape[1]
    temp = nt - Par["lb"] - Par["LF"] + 1
    x_idx = np.arange(temp).reshape(-1, 1)
    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx + i)
    x_idx = np.concatenate(x_idx_ls, axis=1)
    t_idx = np.arange(Par["lf"]).reshape(-1)
    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=Par["LF"])
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    tmp_lr = paddle.optimizer.lr.LambdaDecay(
        lr_lambda=lr_lambda, last_epoch=last_epoch, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    return tmp_lr


def rollout(model, x, t, NT, Par, batch_size):
    y_pred_ls = []
    bs = batch_size
    end = bs

    while True:
        start = end - bs
        out_ls = []

        if start >= x.shape[0]:
            break
        temp_x1 = x[start:end]
        out_ls = [temp_x1]
        traj = paddle.concat(x=out_ls, axis=1)

        while traj.shape[1] < NT:
            with paddle.no_grad():
                temp_x = paddle.repeat_interleave(x=temp_x1, repeats=Par["lf"], axis=0)
                temp_t = t.tile(repeat_times=traj.shape[0])
                with paddle.amp.auto_cast(enable=False):
                    out = model(temp_x, temp_t).reshape([-1, Par["lf"], Par["nf"], Par["nx"], Par["ny"]])
                out_ls.append(out)
                traj = paddle.concat(x=out_ls, axis=1)
                temp_x1 = traj[:, -Par["lb"] :]
        pred = paddle.concat(x=out_ls, axis=1)[:, Par["lb"] : NT]
        y_pred_ls.append(pred)
        end = end + bs
        if end - bs > x.shape[0] + 1:
            break

    if len(y_pred_ls) > 0:
        y_pred = paddle.concat(y_pred_ls, axis=0)
    else:
        y_pred = paddle.zeros([0, NT - Par["lb"], Par["nf"], Par["nx"], Par["ny"]])
    return y_pred


if __name__ == "__main__":
    seed_value = 23
    data_type = "float32"
    save_dir = f"seed_{seed_value}"
    logger = init_all(seed_value, name=save_dir, dtype=data_type)
    scaler = paddle.amp.GradScaler(incr_every_n_steps=2000, init_loss_scaling=65536.0)
    begin_time = time.time()
    data_dir = "../data/UX_nan_filtered.npy"
    logger.info(f"Data path: {data_dir}")
    traj = np.load(data_dir)
    traj = np.expand_dims(traj, axis=0)
    bad_timesteps = []
    for t in range(traj.shape[1]):
        frame = traj[0, t]
        if np.abs(frame).max() > 10.0:
            logger.info(f"! Time step {t} has extreme values: max={np.abs(frame).max():.2e}")
            bad_timesteps.append(t)
    if bad_timesteps:
        logger.info(f"Bad time steps: {bad_timesteps}")
        sys.exit()
    mask_dir = "../data/mask.npy"
    logger.info(f"Mask path: {mask_dir}")
    mask = np.load(mask_dir).reshape(1, 1, traj.shape[-2], traj.shape[-1])
    traj = traj * mask
    traj = np.expand_dims(traj, axis=2)
    logger.info(f"Data loading time: {time.time() - begin_time:.2f}s")
    traj_train = traj[:, :800]
    traj_val = traj[:, 800:900]
    traj_test = traj[:, 900:]

    logger.info(f"Shape of whole data (traj): {traj.shape}")
    logger.info(f"Shape of train data (traj_train): {traj_train.shape}")
    logger.info(f"Shape of val data (traj_val): {traj_val.shape}")
    logger.info(f"Shape of test data (traj_test): {traj_test.shape}\n")

    Par = {}
    Par["nx"] = traj_train.shape[-2]
    Par["ny"] = traj_train.shape[-1]
    Par["nf"] = 1
    Par["d_emb"] = 128

    logger.info(f"Dimension of flow (nx*ny): ({Par['nx']}, {Par['ny']})")
    logger.info(f"Number of features (nf): {Par['nf']}")

    Par["lb"] = 10
    Par["lf"] = 2
    Par["LF"] = 10
    Par["channels"] = Par["nf"] * Par["lb"]
    Par["num_epochs"] = 500
    logger.info(f"Number of timesteps as inputs (lb): {Par['lb']}")
    logger.info(f"Number of timesteps as outputs (lf): {Par['lf']}")
    logger.info(f"Number of timesteps for long-term prediction (LF): {Par['LF']}")
    logger.info(f"Number epochs: {Par['num_epochs']}\n")

    time_cond = np.linspace(0, 1, Par["lf"])
    if Par["lf"] == 1:
        time_cond = np.linspace(0, 1, Par["lf"]) + 1

    t_min = np.min(time_cond)
    t_max = np.max(time_cond)
    if Par["lf"] == 1:
        t_min = 0
        t_max = 1

    Par["inp_shift"] = float(np.mean(traj_train))
    Par["inp_scale"] = float(np.std(traj_train))
    Par["out_shift"] = float(np.mean(traj_train))
    Par["out_scale"] = float(np.std(traj_train))
    Par["t_shift"] = float(t_min)
    Par["t_scale"] = float(t_max - t_min)
    Par["time_cond"] = time_cond.tolist()
    logger.info(f"Input shift of trai_train: {Par['inp_shift']}")
    logger.info(f"Input scale of trai_train: {Par['inp_scale']}")
    logger.info(f"Output shift of trai_train: {Par['out_shift']}")
    logger.info(f"Output scale of trai_train: {Par['out_scale']}")
    logger.info(f"Time shift: {Par['t_shift']}")
    logger.info(f"Time scale: {Par['t_scale']}")
    logger.info(f"Time cond: {Par['time_cond']}\n")

    Par["mask"] = paddle.to_tensor(mask, dtype=data_type)

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, paddle.Tensor)):
            return obj.tolist()
        return obj

    with open("Par.json", "w") as f:
        json.dump(Par, f, default=convert_to_serializable)

    traj_train_tensor = paddle.to_tensor(data=traj_train, dtype=data_type)
    traj_val_tensor = paddle.to_tensor(data=traj_val, dtype=data_type)
    traj_test_tensor = paddle.to_tensor(data=traj_test, dtype=data_type)
    time_cond_tensor = paddle.to_tensor(data=time_cond, dtype=data_type)
    begin_time = time.time()
    x_idx_train, t_idx_train, y_idx_train = preprocess_train(traj_train, Par)
    logger.info("Shape of train dataset")
    logger.info(f"x_idx_train: {x_idx_train.shape}")
    logger.info(f"t_idx_train: {t_idx_train.shape}")
    logger.info(f"y_idx_train: {y_idx_train.shape}\n")
    x_idx_val, t_idx_val, y_idx_val = preprocess(traj_val, Par)
    logger.info("Shape of val dataset")
    logger.info(f"x_idx_val: {x_idx_val.shape}")
    logger.info(f"t_idx_val: {t_idx_val.shape}")
    logger.info(f"y_idx_val: {y_idx_val.shape}\n")
    x_idx_test, t_idx_test, y_idx_test = preprocess(traj_test, Par)
    logger.info("Shape of test dataset")
    logger.info(f"x_idx_test: {x_idx_test.shape}")
    logger.info(f"t_idx_test: {t_idx_test.shape}")
    logger.info(f"y_idx_test: {y_idx_test.shape}\n")
    logger.info(f"Data preprocess time: {time.time() - begin_time:.2f}s\n")
    train_dataset = YourDataset_train(x_idx_train, t_idx_train, y_idx_train)
    val_dataset = YourDataset(x_idx_val, y_idx_val)
    test_dataset = YourDataset(x_idx_test, y_idx_test)

    train_batch_size = 20  # 100
    val_batch_size = 20  # 100
    test_batch_size = 20  # 100
    logger.info(f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}")
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=val_batch_size)
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    model = Unet2D(
        dim=16,
        Par=Par,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
    ).astype("float32")

    criterion = CustomLoss(Par)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=5 * 1e-05, weight_decay=1e-06)
    scheduler = combined_scheduler(
        optimizer,
        Par["num_epochs"] * len(train_loader),
        int(0.1 * Par["num_epochs"]) * len(train_loader),
    )

    # Training loop
    num_epochs = Par["num_epochs"]
    best_val_loss = float("inf")
    best_model_id = 0
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        begin_time = time.time()
        model.train()
        train_loss = 0.0
        for x_idx, t_idx, y_idx in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            x = traj_train_tensor[0, x_idx]
            t = time_cond_tensor[t_idx]
            y_true = traj_train_tensor[0, y_idx]
            optimizer.clear_gradients(set_to_zero=False)

            with paddle.amp.auto_cast(enable=False):
                y_pred = model(x, t)
                loss = criterion(y_pred, y_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            scheduler.step()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with paddle.no_grad():
            for x_idx, y_idx in val_loader:
                x = traj_val_tensor[0, x_idx]
                t = time_cond_tensor[t_idx_val]
                y_true = traj_val_tensor[0, y_idx]
                y_pred = rollout(model, x, t, Par["lb"] + Par["LF"], Par, val_batch_size)
                with paddle.amp.auto_cast(enable=False):
                    loss = criterion(y_pred, y_true)
                val_loss += loss.item()
            make_plot(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), epoch)
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_id = epoch
            paddle.save(obj=model.state_dict(), path="models/best_model.pdparams")
        elapsed_time = time.time() - begin_time
        logger.info(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, "
            f"Best model: {best_model_id}, Learning rate: {scheduler.get_lr():.4e}, "
            f"Epoch time: {elapsed_time:.2f}"
        )
    logger.info("Training finished.")
    logger.info(f"Training Time: {time.time() - t0:.1f}s")
    model.eval()
    test_loss = 0.0
    with paddle.no_grad():
        for x_idx, y_idx in test_loader:
            x = traj_test_tensor[0, x_idx]
            t = time_cond_tensor[t_idx_test]
            y_true = traj_test_tensor[0, y_idx]
            y_pred = rollout(model, x, t, Par["lb"] + Par["LF"], Par, val_batch_size)
            with paddle.amp.auto_cast(enable=False):
                loss = criterion(y_pred, y_true)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.4e}")
