import copy
import math
import os
import re
import traceback
from copy import deepcopy
from itertools import combinations
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import paddle
import sympy as sp
import tqdm

from .._paths import oracle_model_path
from ..symbolicregression.envs.generators import Node
from ..symbolicregression.metrics import compute_metrics
from ..symbolicregression.model import utils_wrapper


class SimpleNet(paddle.nn.Module):
    def __init__(self, _in):
        super().__init__()
        self.linear1 = paddle.compat.nn.Linear(_in, 128)
        self.linear2 = paddle.compat.nn.Linear(128, 128)
        self.linear3 = paddle.compat.nn.Linear(128, 64)
        self.linear4 = paddle.compat.nn.Linear(64, 64)
        self.linear5 = paddle.compat.nn.Linear(64, 1)

    def forward(self, x):
        x = paddle.tanh(self.linear1(x))
        x = paddle.tanh(self.linear2(x))
        x = paddle.tanh(self.linear3(x))
        x = paddle.tanh(self.linear4(x))
        x = self.linear5(x)
        return x


class Oracle:
    def __init__(
        self,
        env,
        generator,
        params,
        oracle_scale=None,
        minimum_sep_cutoff_coeff=2,
        maximum_sep_cutoff_coeff=10,
        use_merge_inference=True,
        use_cache_sep=True,
    ):
        self.env = env
        self.generator = generator
        self.params = params
        self.oracle_scale = oracle_scale
        self.minimum_sep_cutoff_coeff = minimum_sep_cutoff_coeff
        self.maximum_sep_cutoff_coeff = maximum_sep_cutoff_coeff
        self.use_merge_inference = use_merge_inference
        self.use_cache_sep = use_cache_sep
        self.multi_sep_cache_idx = []
        self.multi_sep_cache_group = []

    def group_bit_to_group(self, bit):
        bit = int(bit)
        g = []
        while bit > 0:
            lowest_bit = bit & -bit
            index = int(np.log2(lowest_bit))
            g.append(index)
            bit &= bit - 1
        return np.array(g)

    def group_to_group_bit(self, group):
        assert isinstance(group, np.ndarray)
        bit = 0
        for g in group:
            bit |= 1 << g
        return bit

    def group_to_str(self, group):
        assert isinstance(group, list) and isinstance(group[0], list)
        s = "|".join([",".join(list(map(lambda x: str(int(x)), g))) for g in group])

    def retrieve_oracle_types(self):
        return ["original", "oracle"]

    def sample_training_datapoint(self, num_variables, val_fn, use_maxi_y=True):
        num_to_sample = 100000
        num_sampled = 0
        xs = np.zeros((0, num_variables))
        ys = np.zeros((0, 1))
        max_trial = 5000
        current_trial = 0
        median_y = None
        while num_sampled < num_to_sample and current_trial < max_trial:
            x = np.random.uniform(0, 8, (num_to_sample, num_variables))
            y = val_fn(x) + np.random.normal(0, 0, (x.shape[0],))
            y = y.reshape((-1, 1))
            idx_nans = np.isnan(y).reshape(-1)
            x = x[~idx_nans, :]
            y = y[~idx_nans, :]
            if median_y is None:
                median_y = max(0, np.median(np.abs(y)))
            idx_outliers = (np.abs(y) > 10 * min(median_y, 10000000.0)).reshape(-1)
            x = x[~idx_outliers, :]
            y = y[~idx_outliers, :]
            num_sampled += y.shape[0]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            current_trial += 1
        if xs.shape[0] < num_to_sample:
            return None, None, None
        xs = xs[:num_to_sample]
        ys = ys[:num_to_sample]
        try:
            gamma = self.params.eval_noise_gamma
            if self.params.eval_noise_type == "multiplicative":
                norm = np.linalg.norm((np.abs(ys) + 1e-100) / np.sqrt(ys.shape[0]))
                noise = gamma * norm * np.random.randn(*ys.shape)
            elif self.params.eval_noise_type == "additive":
                noise = gamma * ys * np.random.randn(*ys.shape)
            ys += noise
        except Exception as e:
            print(e, "norm computation error")
            return
        if use_maxi_y:
            maxi_y = np.max(np.abs(ys))
        else:
            maxi_y = np.array([1])
        ys /= maxi_y
        return xs, ys, maxi_y, median_y

    def train_oracle(
        self,
        xy,
        epochs=500,
        batch_size=2048,
        lr=0.01,
        test_rate=0,
        test_freq=-1,
        N_reg_lr=4,
        plot=False,
        device="cpu",
        tqdm_bar=True,
    ):
        x, y = xy
        if isinstance(x, np.ndarray):
            X = paddle.from_numpy(x).float()
            Y = paddle.from_numpy(y).float()
        else:
            X = x
            Y = y
        dataset = paddle.io.TensorDataset([X, Y])
        num_samples = len(dataset)
        train_size = int(num_samples * (1 - test_rate))
        test_size = num_samples - train_size
        train_dataset, test_dataset = paddle.io.random_split(
            dataset, [train_size, test_size]
        )
        train_loader = paddle.io.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        if test_rate > 0:
            test_loader = paddle.io.DataLoader(
                dataset=test_dataset, batch_size=len(test_dataset)
            )
        model = SimpleNet(x.shape[1]).to(device)

        def loss_fn(pred, targ):
            denom = targ**2
            denom = paddle.sqrt(denom.sum() / len(denom)) + 1e-10
            return (
                paddle.sqrt(paddle.nn.functional.mse_loss(input=pred, label=targ))
                / denom
            )

        mses = []
        pbar = tqdm.tqdm(total=N_reg_lr * epochs) if tqdm_bar else None
        for i in range(N_reg_lr):
            # PaddlePaddle: 显式使用list()确保参数格式正确
            optimizer = paddle.optimizer.Adam(
                parameters=list(model.parameters()), learning_rate=lr, weight_decay=0.0
            )
            for epoch in range(1, epochs + 1):
                model.train()
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pred = model(x_batch)
                    if paddle.any(paddle.isnan(pred)) or paddle.any(paddle.isnan(loss)):
                        a = 1
                if test_freq > 0 and epoch % test_freq == 0:
                    model.eval()
                    mse = 0
                    with paddle.no_grad():
                        for x_batch, y_batch in test_loader:
                            x_batch = x_batch.to(device)
                            y_batch = y_batch.to(device)
                            pred = model(x_batch)
                            loss = loss_fn(pred, y_batch)
                            mse += loss.cpu().detach().numpy()
                    mses.append(mse)
                if tqdm_bar:
                    pbar.update(1)
            lr /= 5
        if plot:
            plt.plot(mses)
            plt.show()
        model.eval()
        mse = 0
        if test_rate > 0:
            with paddle.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)
                    mse += loss.cpu().detach().numpy()
        else:
            with paddle.no_grad():
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)
                    mse += loss.cpu().detach().numpy()
        return model, mse

    def sample_fixed_x(self, model, num_variables, median_y, fixed_x_range=None):
        trials = 0
        max_trials = 5
        if median_y is None:
            median_y = 100000000.0
        while trials < max_trials:
            if fixed_x_range is not None:
                fixed_x = np.random.uniform(
                    fixed_x_range[0], fixed_x_range[1], (100, num_variables)
                )
            elif trials > 0:
                fixed_x = np.random.uniform(0.5, 2 + trials, (100, num_variables))
            else:
                fixed_x = np.random.uniform(1, 2, (100, num_variables))
            fixed_x_tensor = (
                paddle.from_numpy(fixed_x).float().to(device=self.params.device)
            )
            tar_y = model(fixed_x_tensor).detach().cpu().numpy().reshape(-1)
            idx_nan = np.abs(tar_y) > 8 * min(median_y, 10000000.0)
            tar_y = tar_y[~idx_nan]
            fixed_x = fixed_x[~idx_nan]
            if len(fixed_x) >= 5:
                sorted_indices = np.argsort(np.abs(tar_y))
                median_index = sorted_indices[len(tar_y) // 2]
                fixed_x = fixed_x[median_index]
                break
            trials += 1
        if trials == max_trials:
            raise OverflowError()
        return fixed_x

    def differentiate(
        self, model, x, apply_func=["id", "log"], sample_type="rand", variable_idx=0
    ):
        num_sample = 1000
        num_variables = x.shape[1]
        if sample_type == "seq":
            assert False
            test_set = np.zeros((num_sample, num_variables))
            test_set[:] = np.random.uniform(-3, 3, num_variables)
            test_set[:, variable_idx] = np.linspace(-10, 10, num_sample).reshape(-1)
        elif sample_type == "rand":
            test_set = np.random.uniform(1 / 2, 3, (num_sample, num_variables))
        elif sample_type == "original":
            test_set = x.copy()[:num_sample]
        else:
            raise ValueError()
        xx = paddle.from_numpy(test_set).float().to(device=self.params.device)
        fs_pred = model(xx).detach().cpu().clone()
        gs_pred = paddle.zeros((0, num_variables))
        for x in test_set:
            xx = (
                paddle.from_numpy(x)
                .float()
                .reshape((1, num_variables))
                .to(device=self.params.device)
            )
            xx.stop_gradient = not True
            y = model(xx)
            y.backward()
            g_pred = xx.grad.detach().cpu().clone().unsqueeze(0).squeeze(0)
            gs_pred = paddle.cat((gs_pred, g_pred), dim=0)
        hs_pred = paddle.zeros((0, num_variables, num_variables))
        for x in test_set:
            # 确保输入张量可求导（PaddlePaddle: 必须设置以计算二阶导数）
            xx = paddle.from_numpy(x).float().to(device=self.params.device)
            xx.stop_gradient = False

            # 创建 Hessian 对象
            h_pred_obj = paddle.incubate.autograd.Hessian(
                func=model, xs=xx, is_batched=False
            )
            # PaddlePaddle: 使用切片操作提取实际的张量矩阵
            h_pred = h_pred_obj[:]

            # 标准张量操作（与 PyTorch 版本保持一致）
            h_pred = h_pred.detach().cpu().clone().unsqueeze(0)
            hs_pred = paddle.cat((hs_pred, h_pred), dim=0)
        res = []
        for func in apply_func:
            if func == "id":
                res.append(paddle.compat.median(paddle.abs(hs_pred), dim=0)[0])
            elif func == "log":
                hs_log_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_log_1 = -1 / f**2 * g * g.T + 1 / f * h
                    h_log_2 = -g * g.T + f * h
                    h_log = paddle.where(
                        paddle.abs(h_log_1) < paddle.abs(h_log_2), h_log_1, h_log_2
                    ).unsqueeze(0)
                    hs_log_pred = paddle.cat((hs_log_pred, h_log), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_log_pred), dim=0)[0])
            elif func == "pow2":
                hs_pow2_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_pow2 = 2 * g * g.T + 2 * f * h
                    h_pow2 = h_pow2.unsqueeze(0)
                    hs_pow2_pred = paddle.cat((hs_pow2_pred, h_pow2), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_pow2_pred), dim=0)[0])
            elif func == "pow2-log":
                if "log" in apply_func:
                    hs_pow2_log_pred = hs_log_pred.clone()
                else:
                    assert False
                res.append(paddle.compat.median(paddle.abs(hs_pow2_log_pred), dim=0)[0])
            elif func == "sin":
                hs_sin_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_sin = -paddle.sin(f) * g * g.T + paddle.cos(f) * f * h
                    h_sin = h_sin.unsqueeze(0)
                    hs_sin_pred = paddle.cat((hs_sin_pred, h_sin), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_sin_pred), dim=0)[0])
            elif func == "sin-log":
                hs_sin_log_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_sin_log_1 = (
                        -(
                            paddle.cos(paddle.log(paddle.abs(f) + 1e-10))
                            + paddle.sin(paddle.log(paddle.abs(f) + 1e-10))
                        )
                        / f**2
                        * g
                        * g.T
                        + paddle.cos(paddle.log(paddle.abs(f) + 1e-10)) / f * h
                    )
                    h_sin_log_2 = (
                        -(
                            paddle.cos(paddle.log(paddle.abs(f) + 1e-10))
                            + paddle.sin(paddle.log(paddle.abs(f) + 1e-10))
                        )
                        * g
                        * g.T
                        + paddle.cos(paddle.log(paddle.abs(f) + 1e-10)) * f * h
                    )
                    h_sin_log = paddle.where(
                        paddle.abs(h_sin_log_1) < paddle.abs(h_sin_log_2),
                        h_sin_log_1,
                        h_sin_log_2,
                    ).unsqueeze(0)
                    hs_sin_log_pred = paddle.cat((hs_sin_log_pred, h_sin_log), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_sin_log_pred), dim=0)[0])
            elif func == "cos":
                hs_cos_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_cos = -paddle.cos(f) * g * g.T - paddle.sin(f) * f * h
                    h_cos = h_cos.unsqueeze(0)
                    hs_cos_pred = paddle.cat((hs_cos_pred, h_cos), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_cos_pred), dim=0)[0])
            elif func == "cos-log":
                hs_cos_log_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_cos_log_1 = (
                        -(
                            paddle.cos(paddle.log(paddle.abs(f) + 1e-10))
                            - paddle.sin(paddle.log(paddle.abs(f) + 1e-10))
                        )
                        / f**2
                        * g
                        * g.T
                        - paddle.sin(paddle.log(paddle.abs(f) + 1e-10)) / f * h
                    )
                    h_cos_log_2 = (
                        -(
                            paddle.cos(paddle.log(paddle.abs(f) + 1e-10))
                            - paddle.sin(paddle.log(paddle.abs(f) + 1e-10))
                        )
                        * g
                        * g.T
                        - paddle.sin(paddle.log(paddle.abs(f) + 1e-10)) * f * h
                    )
                    h_cos_log = paddle.where(
                        paddle.abs(h_cos_log_1) < paddle.abs(h_cos_log_2),
                        h_cos_log_1,
                        h_cos_log_2,
                    ).unsqueeze(0)
                    hs_cos_log_pred = paddle.cat((hs_cos_log_pred, h_cos_log), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_cos_log_pred), dim=0)[0])
            elif func == "inv":
                hs_inv_pred = paddle.zeros(0, num_variables, num_variables)
                for h, g, f in zip(hs_pred, gs_pred, fs_pred):
                    g = g.reshape((-1, 1))
                    h_inv_1 = 2 / f**3 * g * g.T - 1 / f**2 * h
                    h_inv_2 = 2 * g * g.T - f * h
                    h_inv = paddle.where(
                        paddle.abs(h_inv_1) < paddle.abs(h_inv_2), h_inv_1, h_inv_2
                    ).unsqueeze(0)
                    hs_inv_pred = paddle.cat((hs_inv_pred, h_inv), dim=0)
                res.append(paddle.compat.median(paddle.abs(hs_inv_pred), dim=0)[0])
            elif func == "inv-log":
                if "log" in apply_func:
                    hs_inv_log_pred = hs_log_pred.clone()
                else:
                    assert False
                res.append(paddle.compat.median(paddle.abs(hs_inv_log_pred), dim=0)[0])
            else:
                raise ValueError()
        return res

    def seperate_structure(self, sep, num_variables, use_cache=False):
        if not self.use_cache_sep or self.multi_sep_cache_idx == [] or not use_cache:
            group = [np.arange(num_variables)]
            sep_idx = sep
        else:
            assert len(sep) == len(self.multi_sep_cache_idx) + 1
            group = self.multi_sep_cache_group
            sep_idx = [sep[-1]]
        possible_delete_idx = set()
        for idx in sep_idx:
            for i in range(len(group)):
                idx0 = group[i] == idx[0]
                idx1 = group[i] == idx[1]
                if any(idx0) and any(idx1):
                    group0 = group[i][~idx0]
                    group1 = group[i][~idx1]
                    group[i] = group0
                    group.append(group1)
                    possible_delete_idx.add(i)
                    possible_delete_idx.add(len(group) - 1)
        idx_to_delete = []
        for i in range(len(group)):
            if i not in possible_delete_idx:
                continue
            for j in range(len(group)):
                if i == j:
                    continue
                if all([(ele in group[j]) for ele in group[i]]):
                    if len(group[i]) == len(group[j]):
                        idx_to_delete.append(max(i, j))
                    else:
                        idx_to_delete.append(i)
        res_group = [group[i] for i in range(len(group)) if i not in idx_to_delete]
        self.multi_sep_cache_idx = copy.deepcopy(sep)
        self.multi_sep_cache_group = copy.deepcopy(res_group)
        return res_group

    def oracle_seperate(self, diffs, mse, use_oracle_types):
        assert len(use_oracle_types) == len(diffs)
        num_variables = diffs[0].shape[0]
        sep_idxs = []
        groups = []
        sep_type = []
        for diff, oracle_type in zip(diffs, use_oracle_types):
            sep_idxs.append([])
            groups.append([np.arange(num_variables)])
            sep_type.append(oracle_type)
            if num_variables == 1:
                continue
            diff = [
                (diff[i][j], (i, j))
                for i in range(num_variables)
                for j in range(i + 1, num_variables)
            ]
            diff.sort(key=lambda x: x[0])
            minimum_sep = diff[0][0]
            minimum_sep_cutoff = minimum_sep * self.minimum_sep_cutoff_coeff
            maximum_sep_cutoff = minimum_sep * self.maximum_sep_cutoff_coeff
            sep_idx_start = max(
                [0]
                + [
                    idx
                    for idx, (diff_value, coord) in enumerate(diff)
                    if diff_value < minimum_sep_cutoff
                ]
            )
            sep_idx_end = max(
                [0]
                + [
                    idx
                    for idx, (diff_value, coord) in enumerate(diff)
                    if diff_value <= max(maximum_sep_cutoff, 1)
                ]
            )
            if not self.params.oracle_sep_multinum:
                sep_idx_end = sep_idx_start
            self.multi_sep_cache_idx = []
            self.multi_sep_cache_group = []
            for i in range(sep_idx_start, sep_idx_end + 1):
                sep_idx = [
                    coord for idx, (diff_value, coord) in enumerate(diff) if idx <= i
                ]
                sep_idxs.append(sep_idx)
                groups.append(
                    self.seperate_structure(sep_idx, num_variables, use_cache=True)
                )
                sep_type.append(oracle_type)
        return groups, sep_type, sep_idxs

    def merge_groups(self, sep_groups, sep_types):
        merged_groups = []
        merged_types = []
        for sep_group, sep_type in zip(sep_groups, sep_types):
            _fn, _log = sep_type.split(",")
            for single_group in sep_group:
                group_bit = self.group_to_group_bit(single_group)
                group_bit = _fn + "," + str(group_bit)
                if group_bit not in merged_groups:
                    merged_groups.append(group_bit)
                    merged_types.append(_fn)
        return merged_groups, merged_types

    def sample_oracle_datapoint(
        self, model, x, y, median_y, fixed_x, sep_groups, sep_types
    ):
        params = self.params
        if median_y is None:
            median_y = 100000000.0
        new_xs = []
        new_ys = []
        num_variables = x.shape[1]
        for sep_group, sep_type in zip(sep_groups, sep_types):
            _fn, group_bit = sep_group.split(",")
            single_group = self.group_bit_to_group(group_bit)
            if len(single_group) == num_variables:
                new_x = x[: params.max_input_points].copy()
                new_x_tensor = (
                    paddle.from_numpy(new_x).float().to(device=self.params.device)
                )
                new_y = model(new_x_tensor).detach().cpu().numpy().astype(y.dtype)
                outlier_idx = (np.abs(new_y) > 8 * min(median_y, 10000000.0)).reshape(
                    -1
                )
                new_new_x = new_x[~outlier_idx][
                    : params.max_input_points * max(1, params.max_number_bags)
                ]
                new_new_y = new_y[~outlier_idx][
                    : params.max_input_points * max(1, params.max_number_bags)
                ]
            else:
                idx_to_choose = single_group
                idx_not_choose = np.array(
                    [k for k in range(num_variables) if k not in single_group]
                )
                new_x = x[: params.max_input_points].copy()
                new_x[:, idx_not_choose] = fixed_x[idx_not_choose]
                new_x_tensor = (
                    paddle.from_numpy(new_x).float().to(device=self.params.device)
                )
                new_y = model(new_x_tensor).detach().cpu().numpy().astype(y.dtype)
                new_x = x[: params.max_input_points, idx_to_choose].copy()
                outlier_idx = (np.abs(new_y) > 8 * min(median_y, 10000000.0)).reshape(
                    -1
                )
                new_new_x = new_x[~outlier_idx][
                    : params.max_input_points * max(1, params.max_number_bags)
                ]
                new_new_y = new_y[~outlier_idx][
                    : params.max_input_points * max(1, params.max_number_bags)
                ]
            if _fn == "id":
                pass
            elif _fn == "inv":
                new_new_y = 1 / new_new_y
            elif _fn == "sqrt":
                new_new_y = new_new_y**2
            elif _fn == "arcsin":
                new_new_y = np.sin(new_new_y)
            elif _fn == "arccos":
                new_new_y = np.cos(new_new_y)
            else:
                raise ValueError
            new_xs.append(new_new_x)
            new_ys.append(new_new_y)
        return new_xs, new_ys

    def sample_oracle_hints(self, hints):
        units = copy.deepcopy(hints[0])
        res_hints = []
        for i in range(len(hints)):
            current_hint = [
                deepcopy(
                    hints[i][self.single_expr_idx_to_original_expr_idx[single_expr_idx]]
                )
                for single_expr_idx in range(len(self.merged_groups))
            ]
            if i == 0:
                for single_expr_idx in range(len(self.merged_groups)):
                    original_expr_idx = self.single_expr_idx_to_original_expr_idx[
                        single_expr_idx
                    ]
                    _fn, group_bit = self.merged_groups[single_expr_idx].split(",")
                    current_hint[single_expr_idx] = [
                        units[original_expr_idx][k]
                        for k in self.group_bit_to_group(group_bit)
                    ]
                    current_hint[single_expr_idx] += [
                        deepcopy(units[original_expr_idx][-1])
                    ]
                    if _fn == "sqrt":
                        current_hint[single_expr_idx][-1] *= 2
                    elif _fn == "inv":
                        current_hint[single_expr_idx][-1] *= -1
            elif i == len(hints) - 1:
                for single_expr_idx in range(len(self.merged_groups)):
                    original_expr_idx = self.single_expr_idx_to_original_expr_idx[
                        single_expr_idx
                    ]
                    _fn, group_bit = self.merged_groups[single_expr_idx].split(",")
                    current_hint[single_expr_idx] += [
                        [
                            self.oracle_consts[original_expr_idx][k],
                            units[original_expr_idx][k],
                        ]
                        for k in range(self.original_xs[original_expr_idx].shape[1])
                        if k not in self.group_bit_to_group(group_bit)
                    ]
            res_hints.append(current_hint)
        return res_hints

    def oracle_fit(
        self,
        xs,
        ys,
        expr_idxs,
        hints,
        trees=None,
        name="test",
        verbose=False,
        use_parallel=False,
        use_maxi_y=True,
        epochs=100,
        batch_size=2048,
        lr=0.01,
        fixed_x_range=None,
        use_seperate_type=["id"],
        oracle_file=None,
        N_reg_lr=4,
        save_model=True,
    ):
        if isinstance(xs, np.ndarray):
            xs = [xs]
            ys = [ys]
            expr_idxs = [expr_idxs]
        self.expr_num = len(xs)
        self.original_expr_idx_to_oracle_expr_idx = {}
        self.oracle_expr_idx_to_original_expr_idx = {}
        self.single_expr_idx_to_original_expr_idx = {}
        self.oracle_groups_to_single_expr_idx = []
        self.original_xs = deepcopy(xs)
        self.original_ys = deepcopy(ys)
        self.oracle_consts = []
        self.oracle_sep_idxs = []
        self.oracle_groups = []
        self.oracle_types = []
        self.merged_groups = []
        self.merged_types = []
        res_x = []
        res_y = []
        res_hints = []
        if use_parallel and False:
            pass
        if verbose:
            tqdm_bar = tqdm.tqdm(total=self.expr_num)
        oracle_nums = 0
        single_nums = 0
        for i, (x, y) in enumerate(zip(xs, ys)):
            expr_idx = expr_idxs[i]
            if oracle_file is None:
                filename = oracle_model_path(name, expr_idx)
            elif oracle_file.endswith("/"):
                filename = f"{oracle_file}{name}_{expr_idx}.pth"
            else:
                filename = oracle_file
            if trees is not None:
                use_x, normed_y, maxi_y, median_y = self.sample_training_datapoint(
                    x.shape[1], trees[i].val, use_maxi_y=use_maxi_y
                )
            else:
                if self.oracle_scale == "log":
                    y = np.log(y)
                use_x = x.copy()
                use_y = y.copy()
                if use_maxi_y:
                    maxi_y = np.max(np.abs(use_y))
                else:
                    maxi_y = np.array([1])
                median_y = None
                normed_y = use_y / maxi_y
            if os.path.exists(filename):
                _model = SimpleNet(x.shape[1])
                model_state = paddle.load(path=str(filename))
                _model.load_state_dict(model_state["state_dict"])
                _model = _model.to(device=self.params.device)
                maxi_y = model_state["maxi_y"]
                mse = model_state["loss"]
                median_y = model_state["median_y"]
            else:
                _model, mse = self.train_oracle(
                    xy=(use_x, normed_y),
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    test_rate=0,
                    test_freq=-1,
                    plot=False,
                    device=self.params.device,
                    tqdm_bar=True,
                    N_reg_lr=N_reg_lr,
                )
                model_state = {
                    "state_dict": _model.state_dict(),
                    "maxi_y": maxi_y.item(),
                    "median_y": median_y,
                    "loss": mse,
                }
                if save_model:
                    folder_path = os.path.dirname(filename)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    paddle.save(obj=model_state, path=filename)
                maxi_y = maxi_y.item()
            if self.oracle_scale == "log":
                model = lambda t: paddle.exp(_model(t) * maxi_y)
            else:
                model = lambda t: _model(t) * maxi_y
            fixed_x = self.sample_fixed_x(
                model, x.shape[1], median_y, fixed_x_range=fixed_x_range
            )
            assert "id" in use_seperate_type
            apply_func = []
            use_oracle_type = []
            if "id" in use_seperate_type:
                apply_func.extend(["id", "log"])
                use_oracle_type.extend(["id,add", "id,mul"])
            if "inv" in use_seperate_type:
                apply_func.extend(["inv", "inv-log"])
                use_oracle_type.extend(["inv,add", "inv,mul"])
            if (
                "arccin" in use_seperate_type
                and np.all(hints[0][i][-1] == 0)
                and np.all(y > -np.pi / 2)
                and np.all(y < np.pi / 2)
            ):
                apply_func.extend(["sin", "sin-log"])
                use_oracle_type.extend(["arcsin,add", "arcsin,mul"])
            if (
                "arccos" in use_seperate_type
                and np.all(hints[0][i][-1] == 0)
                and np.all(y > 0)
                and np.all(y < np.pi)
            ):
                apply_func.extend(["cos", "cos-log"])
                use_oracle_type.extend(["arccos,add", "arccos,mul"])
            if "sqrt" in use_seperate_type and np.all(y > 0):
                apply_func.extend(["pow2", "pow2-log"])
                use_oracle_type.extend(["sqrt,add", "sqrt,mul"])
            if trees is not None:
                diffs = self.differentiate(model, use_x, apply_func, sample_type="rand")
            else:
                diffs = self.differentiate(
                    model, use_x, apply_func, sample_type="original"
                )
            groups, sep_types, sep_idxs = self.oracle_seperate(
                diffs, mse, use_oracle_type
            )
            merged_groups, merged_types = self.merge_groups(groups, sep_types)
            new_xs, new_ys = self.sample_oracle_datapoint(
                model, x, y, median_y, fixed_x, merged_groups, merged_types
            )
            self.oracle_consts.append(fixed_x)
            self.oracle_sep_idxs.extend(sep_idxs)
            self.oracle_groups.extend(groups)
            self.oracle_types.extend(sep_types)
            self.merged_groups.extend(merged_groups)
            self.merged_types.extend(merged_types)
            self.original_expr_idx_to_oracle_expr_idx[i] = list(
                range(oracle_nums, oracle_nums + len(groups))
            )
            for oracle_expr_idx in range(oracle_nums, oracle_nums + len(groups)):
                self.oracle_expr_idx_to_original_expr_idx[oracle_expr_idx] = i
            for single_expr_idx in range(single_nums, single_nums + len(merged_groups)):
                self.single_expr_idx_to_original_expr_idx[single_expr_idx] = i
            self.oracle_groups_to_single_expr_idx.append(
                {
                    g_bin: single_expr_idx
                    for g_bin, single_expr_idx in zip(
                        merged_groups,
                        range(single_nums, single_nums + len(merged_groups)),
                    )
                }
            )
            oracle_nums += len(groups)
            single_nums += len(merged_groups)
            res_x.extend(new_xs)
            res_y.extend(new_ys)
            if verbose:
                tqdm_bar.update(1)
        if verbose:
            tqdm_bar.close()
        res_hints = self.sample_oracle_hints(hints)
        return res_x, res_y, res_hints

    def subtract_set(self, sets, set_idx_to_subtract):
        if len(sets) == 1:
            assert set_idx_to_subtract == 0
            return np.array([])
        res = sets[set_idx_to_subtract]
        sets = [sets[i] for i in range(len(sets)) if i != set_idx_to_subtract]
        from functools import reduce

        intersect = reduce(np.intersect1d, sets)
        res = [_res for _res in res if _res not in intersect]
        return res

    def refine(self, X, y, node_to_refine, ignore=[]):
        generator = self.generator
        refinement_strategy = utils_wrapper.BFGSRefinement()
        try:
            formula_skeleton = str(node_to_refine)
            node_skeleton, node_constants = generator.function_to_skeleton(
                node_to_refine, constants_with_idx=True, ignore=ignore, ignore_pow=True
            )
            if self.oracle_scale == "log":
                node_skeleton = Node("log", self.params, children=[node_skeleton])
                y = np.log(y)
            formula_skeleton = str(node_skeleton)
            if formula_skeleton in self.formula_skeleton:
                return copy.deepcopy(self.formula_skeleton[formula_skeleton])
            refined_node, refine_success = refinement_strategy.go(
                env=self.env,
                tree=node_skeleton,
                coeffs0=node_constants,
                X=X,
                y=y,
                downsample=1024,
                stop_after=1,
            )
            if self.oracle_scale == "log":
                refined_node = refined_node.children[0]
            assert refined_node is not None
        except:
            message = traceback.format_exc()
            refined_node = node_to_refine
        if hasattr(self, formula_skeleton):
            self.formula_skeleton[formula_skeleton] = copy.deepcopy(refined_node)
        return refined_node

    def safely_refine(self, X, y, node_to_refine, _fn, safety_types=["id"]):
        refined_nodes = []
        if _fn == "id":
            pass
        elif _fn == "inv":
            y = 1 / y
        elif _fn == "sqrt":
            y = y**2
        elif _fn == "arcsin":
            y = np.sin(y)
        elif _fn == "arccos":
            y = np.cos(y)
        else:
            raise ValueError
        for safely_type in safety_types:
            if safely_type == "id":
                safely_y = y
                root = node_to_refine
                refined_node = self.refine(X, safely_y, root)
                refined_nodes.append(refined_node)
            elif safely_type == "neg":
                safely_y = deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("neg", self.params, children=[safely_node])
                refined_node = self.refine(X, safely_y, root)
                refined_nodes.append(refined_node)
            elif safely_type == "inv":
                safely_y = 1 / deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("inv", self.params, children=[safely_node])
                refined_node = self.refine(X, safely_y, root)
                try:
                    refined_nodes.append(refined_node.children[0])
                except:
                    refined_nodes.append(safely_node)
            elif safely_type == "linear":
                safely_y = deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                child1 = Node("0", self.params)
                child2 = Node("1", self.params)
                child3 = Node("mul", self.params, children=[child2, safely_node])
                root = Node("add", self.params, children=[child1, child3])
                refined_node = self.refine(X, safely_y, root)
                refined_nodes.append(refined_node)
            elif safely_type == "safe":
                safely_y = y
                root = deepcopy(node_to_refine)
                self.generator.unify_const(root)
                self.generator.unify_const3(root)
                refined_node = self.refine(X, safely_y, root, ignore=[0, 1, -1])
                refined_nodes.append(refined_node)
            elif safely_type == "safe-neg":
                safely_y = deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("neg", self.params, children=[safely_node])
                self.generator.unify_const(root)
                self.generator.unify_const3(root)
                refined_node = self.refine(X, safely_y, root, ignore=[0, 1, -1])
                refined_nodes.append(refined_node)
            elif safely_type == "safe-inv":
                safely_y = 1 / deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("inv", self.params, children=[safely_node])
                self.generator.unify_const(root)
                self.generator.unify_const3(root)
                refined_node = self.refine(X, safely_y, root, ignore=[0, 1, -1])
                try:
                    refined_nodes.append(refined_node.children[0])
                except:
                    refined_nodes.append(safely_node)
            else:
                raise ValueError(f"{safely_type} is not permitted")
        refined_nodes = self.order_tree(X, y, refined_nodes)
        res = refined_nodes[0]
        if _fn == "id":
            root = res
        elif _fn == "inv":
            root = Node("inv", self.params, [res])
        elif _fn == "sqrt":
            root = Node("sqrt", self.params, [res])
        elif _fn == "arcsin":
            root = Node("arcsin", self.params, [res])
        elif _fn == "arccos":
            root = Node("arccos", self.params, [res])
        else:
            raise ValueError
        return root

    def evaluate_tree(self, tree, X, y, metric):
        try:
            use_abs = tree.params.use_abs
            tree.params.use_abs = True
            y_tilde = tree.val(X).reshape((-1, 1))
            tree.params.use_abs = use_abs
        except:
            y_tilde = np.array([np.nan]).repeat(X.shape[0])
        if self.oracle_scale == "log":
            metrics = compute_metrics(
                {
                    "true": [np.log(y)],
                    "predicted": [np.log(y_tilde)],
                    "predicted_tree": [tree],
                },
                metrics=metric,
            )
        else:
            metrics = compute_metrics(
                {"true": [y], "predicted": [y_tilde], "predicted_tree": [tree]},
                metrics=metric,
            )
        return metrics[metric][0]

    def order_candidates(self, X, y, candidates, metric="_mse"):
        scores = []
        for candidate in candidates:
            if metric not in candidate:
                score = self.evaluate_tree(candidate["predicted_tree"], X, y, metric)
                if math.isnan(score):
                    score = np.inf if metric.startswith("_") else -np.inf
            else:
                score = candidate[metric]
            candidate[metric] = score
            scores.append(score)
        ordered_idx = np.argsort(scores)
        if not metric.startswith("_"):
            ordered_idx = list(reversed(ordered_idx))
        candidates = [candidates[i] for i in ordered_idx]
        return candidates

    def order_tree(self, X, y, trees, metric="_mse"):
        scores = []
        for tree in trees:
            score = self.evaluate_tree(tree, X, y, metric)
            if math.isnan(score):
                score = np.inf if metric.startswith("_") else -np.inf
            scores.append(score)
        ordered_idx = np.argsort(scores)
        if not metric.startswith("_"):
            ordered_idx = list(reversed(ordered_idx))
        trees = [trees[i] for i in ordered_idx]
        return trees

    def safety_wrapper(self, node):
        dfs_stack = [node]
        while dfs_stack:
            current_node = dfs_stack.pop(0)
            if current_node.value in ["sqrt", "log"]:
                safety_node = Node("abs", self.params, children=current_node.children)
                current_node.children = [safety_node]
            dfs_stack.extend(current_node.children)

    def translate_expr_str(self, expr_tree, groups):
        """
        translate single expr str into oracle expr str,
        i.e. replace operators/variables/consts
        """
        expr_str = []
        for j, e in enumerate(expr_tree):
            e = deepcopy(e)
            self.safety_wrapper(e)
            e = str(e)
            var = groups[j]
            replace_ops = {
                "add": "+",
                "mul": "*",
                "sub": "-",
                "pow": "**",
                "inv": "1/",
                "neg": "-",
                "pi": "3.14159265",
                "E": "2.718281828459045",
            }
            replace_ops.update({f"x_{k}": f"y_{l}" for k, l in enumerate(var)})
            for op, replace_op in replace_ops.items():
                e = e.replace(op, replace_op)
            replace_ops = {f"y_{l}": f"x_{l}" for k, l in enumerate(var)}
            for op, replace_op in replace_ops.items():
                e = e.replace(op, replace_op)
            expr_str.append(e)
        return expr_str

    def _reverse(self, expr_str, groups, consts, oracle_type):
        """
        reverse oracle expr from several single exprs
        """
        _fn, _log = oracle_type.split(",")
        total_expr = "1" if _log == "mul" else "0"
        for k in range(1, len(expr_str) + 1):
            current_expr = []
            for comb in combinations(range(len(expr_str)), k):
                current_group = [groups[_comb] for _comb in comb]
                for t, it in enumerate(comb):
                    evaluation_set = self.subtract_set(current_group, t)
                    _current_expr = expr_str[it]
                    var_dic = {
                        f"x_{_idx}": f"({consts[_idx].item()})"
                        for _idx in evaluation_set
                    }
                    pattern = re.compile("\\b(x_\\d+)\\b")
                    _current_expr = pattern.sub(
                        lambda t: var_dic.get(t.group(1), t.group(0)), _current_expr
                    )
                    current_expr.append(_current_expr)
            current_expr = [
                ("(" + _current_expr + ")") for _current_expr in current_expr
            ]
            if _log == "add":
                current_expr = "+".join(current_expr)
                current_expr = f"+({(-1) ** (k - 1)}/{k} * ({current_expr}))"
            elif _log == "mul":
                current_expr = "*".join(current_expr)
                try:
                    num_expr = sp.parse_expr(current_expr).evalf()
                    if isinstance(num_expr, sp.core.numbers.Float):
                        current_expr = str(sp.Abs(num_expr))
                except:
                    message = traceback.format_exc()
                    pass
                current_expr = f"*({current_expr}) ** ({(-1) ** (k - 1)}/{k})"
            else:
                raise ValueError()
            total_expr += current_expr
        return total_expr

    def reverse(self, original_gens, oracle_gens, eliminate=False):
        res_exprs = []
        self.oracle_exprs = []
        refinement_strategy = self.params.refinement_strategy.split(",")
        for original_expr_idx in range(self.expr_num):
            self.formula_skeleton = {}
            if original_gens[original_expr_idx]["predicted_tree"] is not None:
                original_node = original_gens[original_expr_idx]["predicted_tree"]
                self.generator.unify_const(original_node)
                self.generator.unify_const3(original_node)
                refined_node = self.safely_refine(
                    self.original_xs[original_expr_idx],
                    self.original_ys[original_expr_idx],
                    original_node,
                    "id",
                    refinement_strategy,
                )
            else:
                refined_node = None
            original_expr = {
                "oracle_type": "original",
                "predicted_tree": refined_node,
                "relabed_predicted_tree": refined_node,
                "message": "original",
            }
            res_exprs.append(original_expr)
            oracle_exprs = []
            for oracle_expr_idx in self.original_expr_idx_to_oracle_expr_idx[
                original_expr_idx
            ]:
                consts = self.oracle_consts[original_expr_idx]
                groups = self.oracle_groups[oracle_expr_idx]
                oracle_type = self.oracle_types[oracle_expr_idx]
                _fn, _log = oracle_type.split(",")
                oracle_expr = {"oracle_type": "oracle"}
                expr = [
                    oracle_gens[
                        self.oracle_groups_to_single_expr_idx[original_expr_idx][
                            _fn + "," + str(self.group_to_group_bit(group))
                        ]
                    ]
                    for group in groups
                ]
                expr_tree = [e["predicted_tree"] for e in expr]
                if any([(e is None) for e in expr_tree]):
                    oracle_expr["predicted_tree"] = None
                    oracle_expr["relabed_predicted_tree"] = None
                    oracle_expr["message"] = "some sub-expr are None"
                    oracle_exprs.append(oracle_expr)
                    continue
                expr_str = self.translate_expr_str(expr_tree, groups)
                total_expr = self._reverse(expr_str, groups, consts, oracle_type)
                try:
                    sympy_expr = str(sp.parse_expr(total_expr))
                    node, _ = self.generator.infix_to_node(
                        sympy_expr,
                        label_units=False,
                        sp_parse=False,
                        allow_pow=True,
                        variables=[
                            f"x_{idx}"
                            for idx in range(len(self.oracle_consts[original_expr_idx]))
                        ],
                    )
                except:
                    message = traceback.format_exc()
                    oracle_expr["predicted_tree"] = None
                    oracle_expr["relabed_predicted_tree"] = None
                    oracle_expr["message"] = "complex numbers!"
                    oracle_exprs.append(oracle_expr)
                    continue
                refined_node = self.safely_refine(
                    self.original_xs[original_expr_idx],
                    self.original_ys[original_expr_idx],
                    node,
                    _fn,
                    refinement_strategy,
                )
                oracle_expr["predicted_tree"] = refined_node
                oracle_expr["relabed_predicted_tree"] = refined_node
                oracle_expr[
                    "message"
                ] = f"groups: {'|'.join(','.join(map(str, arr)) for arr in groups)}, type: {oracle_type}"
                oracle_exprs.append(oracle_expr)
            self.oracle_exprs.extend(deepcopy(oracle_exprs))
            ordered_exprs = self.order_candidates(
                self.original_xs[original_expr_idx],
                self.original_ys[original_expr_idx],
                oracle_exprs,
                metric="_mse",
            )
            res_exprs.append(ordered_exprs[0])
            if eliminate:
                original_and_oracle = copy.deepcopy(res_exprs[-2:])
                eliminate_exprs = self.order_candidates(
                    self.original_xs[original_expr_idx],
                    self.original_ys[original_expr_idx],
                    original_and_oracle,
                    metric="_mse",
                )
                res_exprs = res_exprs[:-2]
                res_exprs.append(eliminate_exprs[0])
                res_exprs.append(original_and_oracle[0])
            del self.formula_skeleton
        return res_exprs

    def adjusted_rand_index(self, true_sep, pred_sep, num_variables):
        true_struct = self.seperate_structure(true_sep, num_variables, use_cache=False)
        pred_struct = self.seperate_structure(pred_sep, num_variables, use_cache=False)
        tn, fp, fn, tp = 0, 0, 0, 0
        for i in range(num_variables):
            for j in range(i + 1, num_variables):
                in_true = False
                for group in true_struct:
                    if i in group and j in group:
                        in_true = True
                in_pred = False
                for group in pred_struct:
                    if i in group and j in group:
                        in_pred = True
                if in_true and in_pred:
                    tp += 1
                elif in_true and not in_pred:
                    fn += 1
                elif not in_true and in_pred:
                    fp += 1
                elif not in_true and not in_pred:
                    tn += 1
        if fn == 0 and fp == 0:
            return 1.0
        elif fn == 0 and tp == 0:
            return tn / (tn + fp)
        elif fp == 0 and tn == 0:
            return tp / (tp + fn)
        ari = (
            2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        )
        return ari

    def check_group_subset(self, true_sep, pred_sep, num_variables):
        true_sep = set(map(tuple, true_sep))
        pred_sep = set(map(tuple, pred_sep))
        pred_struct = self.seperate_structure(pred_sep, num_variables, use_cache=False)
        if len(pred_sep) == 0:
            frac = 0
        if true_sep == pred_sep:
            frac = 1
        elif pred_sep.issubset(true_sep):
            frac = len(pred_sep) / len(true_sep)
        else:
            frac = -1
        return frac, pred_struct

    def evaluate_oracle_accuracy(self, trees, is_oracle=True):
        if not is_oracle:
            return [0] * len(trees)
        assert len(trees) == len(self.original_xs)
        aris = []
        res = []
        for oracle_expr_idx in range(len(self.oracle_expr_idx_to_original_expr_idx)):
            original_expr_idx = self.oracle_expr_idx_to_original_expr_idx[
                oracle_expr_idx
            ]
            tree = trees[original_expr_idx]
            num_variables = self.original_xs[original_expr_idx].shape[1]
            sep_idx = self.oracle_sep_idxs[oracle_expr_idx]
            sep_type = self.oracle_types[oracle_expr_idx]
            _fn, _log = sep_type.split(",")
            if _fn == "id":
                sep = self.generator.differentiate(tree, "", num_variables, _log)
            elif _fn == "sqrt":
                root = Node("pow2", self.params, children=[tree])
                sep = self.generator.differentiate(root, "", num_variables, _log)
            elif _fn == "arcsin":
                root = Node("sin", self.params, children=[tree])
                sep = self.generator.differentiate(root, "", num_variables, _log)
            elif _fn == "arccos":
                root = Node("cos", self.params, children=[tree])
                sep = self.generator.differentiate(root, "", num_variables, _log)
            elif _fn == "inv":
                root = Node("inv", self.params, children=[tree])
                sep = self.generator.differentiate(root, "", num_variables, _log)
            else:
                raise ValueError
            ari = self.adjusted_rand_index(sep, sep_idx, num_variables)
            assert -1 <= ari <= 1
            aris.append(ari)
        for original_expr_idx in range(len(self.original_expr_idx_to_oracle_expr_idx)):
            current_res = {
                f"{_fn},{_log}": []
                for _fn in ["id", "inv", "sqrt", "arcsin", "arccos"]
                for _log in ["add", "mul"]
            }
            for oracle_expr_idx in self.original_expr_idx_to_oracle_expr_idx[
                original_expr_idx
            ]:
                oracle_node = self.oracle_exprs[oracle_expr_idx]["predicted_tree"]
                oracle_type = self.oracle_types[oracle_expr_idx]
                X = self.original_xs[original_expr_idx]
                y = self.original_ys[original_expr_idx]
                score = self.evaluate_tree(oracle_node, X, y, metric="r2_zero")
                ari = aris[oracle_expr_idx]
                current_res[oracle_type].append((ari, score))
            res.append(current_res)
        return res
