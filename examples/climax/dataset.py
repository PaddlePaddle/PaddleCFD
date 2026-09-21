import os

import numpy as np
import paddle
import xarray as xr

from normalize import Normalize


def load_x_y(data_path, list_simu, out_var):
    x_all, y_all = {}, {}
    for simu in list_simu:
        input_name = "inputs_" + simu + ".nc"
        output_name = "outputs_" + simu + ".nc"
        if "hist" in simu:
            input_xr = xr.open_dataset(os.path.join(data_path, input_name))
            output_xr = xr.open_dataset(os.path.join(data_path, output_name)).mean(
                dim="member"
            )
            output_xr = output_xr.assign(
                {"pr": output_xr.pr * 86400, "pr90": output_xr.pr90 * 86400}
            ).rename({"lon": "longitude", "lat": "latitude"}).transpose(
                "time", "latitude", "longitude"
            ).drop(["quantile"])
        else:
            input_xr = xr.open_mfdataset(
                [
                    os.path.join(data_path, "inputs_historical.nc"),
                    os.path.join(data_path, input_name),
                ]
            ).compute()
            output_xr = xr.concat(
                [
                    xr.open_dataset(
                        os.path.join(data_path, "outputs_historical.nc")
                    ).mean(dim="member"),
                    xr.open_dataset(os.path.join(data_path, output_name)).mean(
                        dim="member"
                    ),
                ],
                dim="time",
            ).compute()
            output_xr = output_xr.assign(
                {"pr": output_xr.pr * 86400, "pr90": output_xr.pr90 * 86400}
            ).rename({"lon": "longitude", "lat": "latitude"}).transpose(
                "time", "latitude", "longitude"
            ).drop(["quantile"])
        print(input_xr.dims, output_xr.dims, simu)
        x = input_xr.to_array().to_numpy()
        x = x.transpose(1, 0, 2, 3).astype(np.float32)
        x_all[simu] = x
        y = output_xr[out_var].to_array().to_numpy()
        y = y.transpose(1, 0, 2, 3).astype(np.float32)
        y_all[simu] = y
    temp = xr.open_dataset(
        os.path.join(data_path, "inputs_" + list_simu[0] + ".nc")
    ).compute()
    if "latitude" in temp:
        lat = np.array(temp["latitude"])
        lon = np.array(temp["longitude"])
    else:
        lat = np.array(temp["lat"])
        lon = np.array(temp["lon"])
    return x_all, y_all, lat, lon


def input_for_training(x, skip_historical, history, len_historical):
    time_length = x.shape[0]
    if skip_historical:
        X_train_to_return = np.array(
            [
                x[i : i + history]
                for i in range(len_historical - history + 1, time_length - history + 1)
            ]
        )
    else:
        X_train_to_return = np.array(
            [x[i : i + history] for i in range(0, time_length - history + 1)]
        )
    return X_train_to_return


def output_for_training(y, skip_historical, history, len_historical):
    time_length = y.shape[0]
    if skip_historical:
        Y_train_to_return = np.array(
            [
                y[i + history - 1]
                for i in range(len_historical - history + 1, time_length - history + 1)
            ]
        )
    else:
        Y_train_to_return = np.array(
            [y[i + history - 1] for i in range(0, time_length - history + 1)]
        )
    return Y_train_to_return


def split_train_val(x, y, train_ratio=0.9):
    shuffled_ids = np.random.permutation(x.shape[0])
    train_len = int(train_ratio * x.shape[0])
    train_ids = shuffled_ids[:train_len]
    val_ids = shuffled_ids[train_len:]
    return x[train_ids], y[train_ids], x[val_ids], y[val_ids]


class ClimateBenchDataset(paddle.io.Dataset):
    def __init__(
        self, X_train_all, Y_train_all, variables, out_variables, lat, partition="train"
    ):
        super().__init__()
        self.X_train_all = X_train_all
        self.Y_train_all = Y_train_all
        self.len_historical = 165
        self.variables = variables
        self.out_variables = out_variables
        self.lat = lat
        self.partition = partition
        if partition == "train":
            self.inp_transform = self.get_normalize(self.X_train_all)
            self.out_transform = Normalize(np.array([0.0]), np.array([1.0]))
        else:
            self.inp_transform = None
            self.out_transform = None
        if partition == "test":
            self.X_train_all = self.X_train_all[-21:]
            self.Y_train_all = self.Y_train_all[-21:]
            self.get_rmse_normalization()

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 1, 3, 4))
        std = np.std(data, axis=(0, 1, 3, 4))
        return Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize):
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_rmse_normalization(self):
        y_avg = paddle.to_tensor(self.Y_train_all).squeeze(1).mean(0)  # H, W
        w_lat = np.cos(np.deg2rad(self.lat))  # (H,)
        w_lat = w_lat / w_lat.mean()
        w_lat = paddle.to_tensor(w_lat).unsqueeze(-1).astype(y_avg.dtype)  # (H, 1)
        self.y_normalization = paddle.abs(paddle.mean(y_avg * w_lat))

    def __len__(self):
        return self.X_train_all.shape[0]

    def __getitem__(self, index):
        inp = self.inp_transform(paddle.from_numpy(self.X_train_all[index]))
        out = self.out_transform(paddle.from_numpy(self.Y_train_all[index]))
        lead_times = paddle.Tensor([0.0]).to(dtype=inp.dtype)
        return inp, out, lead_times, self.variables, self.out_variables
