import math
import os
import pickle
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from numpy.lib.stride_tricks import sliding_window_view
from paddle import summary

from ppcfd.models.ppdiffusion.matcho import Unet2D


paddle.seed(seed=23)

DTYPE = "float32"

matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"

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

    nt = tuple(traj.shape)[1]
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

    nt = tuple(traj.shape)[1]
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

    scheduler = paddle.optimizer.lr.LambdaDecay(optimizer.get_lr(), lr_lambda, last_epoch)
    optimizer.set_lr_scheduler(scheduler)
    return scheduler


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

        while tuple(traj.shape)[1] < NT:
            with paddle.no_grad():
                temp_x = paddle.repeat_interleave(x=temp_x1, repeats=Par["lf"], axis=0)
                temp_t = t.tile(repeat_times=tuple(traj.shape)[0])
                with paddle.amp.auto_cast(enable=False):
                    out = model(temp_x, temp_t).reshape([-1, Par["lf"], Par["nf"], Par["nx"], Par["ny"]])
                out_ls.append(out)
                traj = paddle.concat(x=out_ls, axis=1)
                temp_x1 = traj[:, -Par["lb"] :]
        pred = paddle.concat(x=out_ls, axis=1)[:, Par["lb"] : NT]
        y_pred_ls.append(pred)
        end = end + bs
        if end - bs > tuple(x.shape)[0] + 1:
            break

    if len(y_pred_ls) > 0:
        y_pred = paddle.concat(y_pred_ls, axis=0)
    else:
        y_pred = paddle.zeros([0, NT - Par["lb"], Par["nf"], Par["nx"], Par["ny"]])
    return y_pred


res = 128
begin_time = time.time()
traj = np.load("../data/UX_nan_filtered.npy")
traj = np.expand_dims(traj, axis=0)
mask = np.load("../data/mask.npy").reshape(1, 1, tuple(traj.shape)[-2], tuple(traj.shape)[-1])
traj = traj * mask
traj = np.expand_dims(traj, axis=2)
print(f"traj: {tuple(traj.shape)}")
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")
traj_train = traj[:, :800]
traj_val = traj[:, 800:900]
traj_test = traj[:, 900:]
Par = {}
Par["nx"] = tuple(traj_train.shape)[-2]
Par["ny"] = tuple(traj_train.shape)[-1]
Par["nf"] = 1
Par["d_emb"] = 128

Par["lb"] = 10
Par["lf"] = 2
Par["LF"] = 10
Par["channels"] = Par["nf"] * Par["lb"]
Par["num_epochs"] = 500

time_cond = np.linspace(0, 1, Par["lf"])
if Par["lf"] == 1:
    time_cond = np.linspace(0, 1, Par["lf"]) + 1

begin_time = time.time()
print("\nTrain Dataset")
x_idx_train, t_idx_train, y_idx_train = preprocess(traj_train, Par)
print("\nValidation Dataset")
x_idx_val, t_idx_val, y_idx_val = preprocess(traj_val, Par)
print("\nTest Dataset")
x_idx_test, t_idx_test, y_idx_test = preprocess(traj_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

t_min = np.min(time_cond)
t_max = np.max(time_cond)
if Par["lf"] == 1:
    t_min = 0
    t_max = 1
Par["inp_shift"] = np.mean(traj_train)
Par["inp_scale"] = np.std(traj_train)
Par["out_shift"] = np.mean(traj_train)
Par["out_scale"] = np.std(traj_train)
Par["t_shift"] = t_min
Par["t_scale"] = t_max - t_min
with open("Par.pkl", "wb") as f:
    pickle.dump(Par, f)

# Create custom datasets
mask_tensor = paddle.to_tensor(data=mask, dtype=DTYPE)
Par["mask"] = mask_tensor

# Create custom datasets
traj_train_tensor = paddle.to_tensor(data=traj_train, dtype=DTYPE)
traj_val_tensor = paddle.to_tensor(data=traj_val, dtype=DTYPE)
traj_test_tensor = paddle.to_tensor(data=traj_test, dtype=DTYPE)
time_cond_tensor = paddle.to_tensor(data=time_cond, dtype=DTYPE)
train_dataset = YourDataset(x_idx_train, y_idx_train)
val_dataset = YourDataset(x_idx_val, y_idx_val)
test_dataset = YourDataset(x_idx_test, y_idx_test)

# Define data loaders
train_batch_size = 20
val_batch_size = 20
test_batch_size = 20
train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=val_batch_size)
test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=test_batch_size)

model = Unet2D(
    dim=16,
    Par=Par,
    dim_mults=(1, 2, 4, 8),
    channels=Par["channels"],
).astype("float32")

path_model = "models/best_model.pdparams"
model.set_state_dict(state_dict=paddle.load(path=str(path_model)))

# summary
input_size = [(1, Par["lb"], Par["nf"], Par["nx"], Par["ny"]), (1,)]
summary(model, input_size=input_size)

# Define loss function and optimizer
criterion = CustomLoss(Par)

# TC-UNet model loss evaluation in training/validation/test sets
# training sets
y_true_ls = []
y_pred_ls = []

model.eval()
train_loss = 0.0
with paddle.no_grad():
    for x_idx, y_idx in train_loader:
        x = traj_train_tensor[0, x_idx]
        t = time_cond_tensor[t_idx_train]
        y_true = traj_train_tensor[0, y_idx]
        y_pred = rollout(model, x, t, Par["lb"] + Par["LF"], Par, train_batch_size)
        with paddle.amp.auto_cast(enable=False):
            loss = criterion(y_pred, y_true)
        train_loss += loss.item()
        y_true_ls.append(y_true.numpy())
        y_pred_ls.append(y_pred.numpy())

train_loss /= len(train_loader)
print(f"Train Loss: {train_loss:.4e}")

TRAIN_TRUE = np.concatenate(y_true_ls, axis=0).reshape(-1, Par["LF"], Par["nx"], Par["ny"]).astype(np.float32)
TRAIN_PRED = np.concatenate(y_pred_ls, axis=0).reshape(-1, Par["LF"], Par["nx"], Par["ny"]).astype(np.float32)
print(f"Train True: {tuple(TRAIN_TRUE.shape)}, Dtype: {TRAIN_TRUE.dtype}")
print(f"Train Pred: {tuple(TRAIN_PRED.shape)}, Dtype: {TRAIN_PRED.dtype}")

# validation sets
y_true_ls = []
y_pred_ls = []

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
        y_true_ls.append(y_true.numpy())
        y_pred_ls.append(y_pred.numpy())

val_loss /= len(val_loader)
print(f"Val Loss: {val_loss:.4e}")
VAL_TRUE = np.concatenate(y_true_ls, axis=0).reshape(-1, Par["LF"], Par["nx"], Par["ny"]).astype(np.float32)
VAL_PRED = np.concatenate(y_pred_ls, axis=0).reshape(-1, Par["LF"], Par["nx"], Par["ny"]).astype(np.float32)
print(f"Val True: {tuple(VAL_TRUE.shape)}, Dtype: {VAL_TRUE.dtype}")
print(f"Val Pred: {tuple(VAL_PRED.shape)}, Dtype: {VAL_PRED.dtype}")

# test sets
y_true_ls = []
y_pred_ls = []

model.eval()
test_loss = 0.0
with paddle.no_grad():
    for x_idx, y_idx in test_loader:
        x = traj_test_tensor[0, x_idx]
        t = time_cond_tensor[t_idx_test]
        y_true = traj_test_tensor[0, y_idx]
        y_pred = rollout(model, x, t, Par["lb"] + Par["LF"], Par, test_batch_size)
        with paddle.amp.auto_cast(enable=False):
            loss = criterion(y_pred, y_true)
        test_loss += loss.item()
        y_true_ls.append(y_true.numpy())
        y_pred_ls.append(y_pred.numpy())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4e}")
TEST_TRUE = np.concatenate(y_true_ls, axis=0).reshape(-1, Par["LF"], Par["nx"], Par["ny"]).astype(np.float32)
TEST_PRED = np.concatenate(y_pred_ls, axis=0).reshape(-1, Par["LF"], Par["nx"], Par["ny"]).astype(np.float32)
print(f"Test True: {tuple(TEST_TRUE.shape)}, Dtype: {TEST_TRUE.dtype}")
print(f"Test Pred: {tuple(TEST_PRED.shape)}, Dtype: {TEST_PRED.dtype}")

# save
np.save("TRAIN_TRUE.npy", TRAIN_TRUE)
np.save("TRAIN_PRED.npy", TRAIN_PRED)
np.save("VAL_TRUE.npy", VAL_TRUE)
np.save("VAL_PRED.npy", VAL_PRED)
np.save("TEST_TRUE.npy", TEST_TRUE)
np.save("TEST_PRED.npy", TEST_PRED)
