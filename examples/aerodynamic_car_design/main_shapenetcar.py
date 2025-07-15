# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import logging
import numpy as np
import paddle
from paddle.io import DataLoader
from tqdm import tqdm
from transolver import Model

from ppcfd.data.shapenetcar_datamodule import GraphDataset
from ppcfd.data.shapenetcar_datamodule import load_train_val_fold


log = logging.getLogger(__name__)
paddle.seed(42)
np.random.seed(42)


def get_nb_trainable_params(model):
    """
    获取模型中的可训练参数数量。

    Args:
        model (torch.nn.Module): 需要计算可训练参数数量的模型。

    Returns:
        int: 模型中的可训练参数数量。

    """
    model_parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    return sum([np.prod(tuple(p.shape)) for p in model_parameters])


def train_epoch(model, train_loader, optimizer, scheduler, reg=1, epoch=0, prof=None):
    """
    当前epoch训练模型

    Args:
        model (paddle.nn.Layer): 模型
        train_loader (paddle.io.DataLoader): 训练数据加载器
        optimizer (paddle.optimizer.Optimizer): 优化器
        scheduler (paddle.optimizer.lr.LearningRateScheduler): 学习率调度器
        reg (float, optional): 正则化项系数，默认为1
        epoch (int, optional): 当前训练轮数，默认为0
        prof (Optional[dict], optional): 用于性能分析的字典，默认为None

    Returns:
        tuple: 包含平均压力损失和平均速度损失的元组

    """
    model.train()
    criterion_func = paddle.nn.MSELoss(reduction="none")
    losses_press = []
    losses_velo = []
    for data in train_loader:
        inputs, targets, surf, _ = data
        bs = inputs.shape[0]
        optimizer.clear_grad()
        out = model(inputs[0])
        p_pred = paddle.stack([out[i][surf[i], -1:] for i in range(bs)], axis=0)
        p_true = paddle.stack([targets[i][surf[i], -1:] for i in range(bs)], axis=0)
        loss_press = criterion_func(p_pred, p_true).mean()
        loss_velo = criterion_func(out[:, :, :-1], targets[:, :, :-1]).mean()
        total_loss = loss_velo + reg * loss_press
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


@paddle.no_grad()
def test(model, test_loader, coef_norm, enable_test=False, eps=1e-8):
    """
    测试模型。

    Args:
        model (paddle.nn.Layer): 需要测试的模型。
        test_loader (paddle.io.DataLoader): 测试数据集加载器。
        coef_norm (list): 归一化系数列表。
        enable_test (bool, optional): 是否启用详细测试输出。默认为 False。

    Returns:
        tuple: 包含以下元素的元组:
            - mean_loss_press (float): 压力损失的均值。
            - mean_loss_velo (float): 速度损失的均值。
            - mean_loss_press_orig (float): 原始压力损失的均值。
            - mean_loss_velo_orig (float): 原始速度损失的均值。
            - spearman_corr (float): Spearman 相关系数。
            - cd_mre (float): 平均相对误差。
    """
    model.eval()
    criterion_func = paddle.nn.MSELoss(reduction="none")
    loss_press_orig, loss_velo_orig = 0.0, 0.0
    losses_press, losses_velo, p_orig, v_orig = [], [], [], []
    for i, data in enumerate(test_loader):
        inputs, targets, surf, sample_name = data
        bs = inputs.shape[0]
        out = model(inputs[0])
        p_pred = paddle.stack([out[j][surf[j], -1:] for j in range(bs)], axis=0)
        p_true = paddle.stack([targets[j][surf[j], -1:] for j in range(bs)], axis=0)
        loss_press = criterion_func(p_pred, p_true).mean()
        loss_velo = criterion_func(out[:, :, :-1], targets[:, :, :-1]).mean()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

        if enable_test is True:
            out_orig = out * (coef_norm[3] + eps) + coef_norm[2]
            targets_orig = targets * (coef_norm[3] + eps) + coef_norm[2]
            p_pred_orig = paddle.stack([out_orig[j][surf[j], -1:] for j in range(bs)], axis=0)
            p_true_orig = paddle.stack([targets_orig[j][surf[j], -1:] for j in range(bs)], axis=0)
            v_pred_orig = paddle.stack([out_orig[j][:, :-1] for j in range(bs)], axis=0)
            v_true_orig = paddle.stack([targets_orig[j][:, :-1] for j in range(bs)], axis=0)
            loss_press_orig = paddle.linalg.norm(p_pred_orig - p_true_orig) / paddle.linalg.norm(p_true_orig)
            loss_velo_orig = paddle.linalg.norm(v_true_orig - v_pred_orig) / paddle.linalg.norm(v_pred_orig)
            log.info(
                f"Test Case {i}, {sample_name[0]}, loss_velo = {loss_velo_orig.item():.4f}, loss_press = {loss_press_orig.item():.4f}"
            )

            # cd_pred = cal_coefficient(Path(args.data_dir) / Path(sample_name[0]), p_pred_orig[0], v_pred_orig[0])
            # cd_true = cal_coefficient(Path(args.data_dir) / Path(sample_name[0]), p_true_orig[0], v_true_orig[0])
            # loss_cd = abs(cd_pred - cd_true) / abs(cd_true)

        p_orig.append(loss_press_orig)
        v_orig.append(loss_velo_orig)
    # spearman_corr = scipy.stats.spearmanr(cd_true_list, cd_pred_list)[0]
    return (
        np.mean(losses_press),
        np.mean(losses_velo),
        np.mean(p_orig),
        np.mean(v_orig),
        0.0,  # spearman_corr
        0.0,  # cd_relative_error
    )


def save_checkpoint(path, epoch, model):
    """
    保存模型检查点。

    Args:
        path (str): 保存检查点的路径。
        epoch (int): 当前训练的轮数。
        model (paddle.Model): 要保存的检查点模型。

    Returns:
        None
    """
    model_path = os.path.join(path, f"model_{epoch}.pdparams")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    paddle.save(state_dict, model_path)


def train(
    config,
    train_dataset,
    val_dataset,
    model,
    path,
    reg=1,
    val_iter=1,
    coef_norm=None,
    enable_test=False,
    enable_prof=False,
):
    """
    训练深度学习模型。

    Args:
        config (dict): 包含训练参数的超参数字典，如学习率（lr）、批量大小（batch_size）和训练轮数（nb_epochs）等。
        train_dataset (Dataset): 训练数据集。
        val_dataset (Dataset): 验证数据集。
        model (Model): 待训练的深度学习模型。
        path (str): 模型和训练结果的保存路径。
        reg (float, optional): 正则化系数，默认为1。
        val_iter (int, optional): 验证间隔，即每训练多少个epoch验证一次，默认为1。
        coef_norm (list, optional): 系数归一化列表，默认为空列表。
        enable_test (bool, optional): 是否启用测试模式，默认为False。
        enable_prof (bool, optional): 是否启用性能分析，默认为False。

    Returns:
        Model: 训练后的模型。

    """
    if coef_norm is None:
        coef_norm = []
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=config.lr, weight_decay=0.0)
    lr_scheduler = paddle.optimizer.lr.OneCycleLR(
        max_learning_rate=config.lr,
        total_steps=(len(train_dataset) // config.batch_size + 1) * config.num_epochs,
        end_learning_rate=config.lr / (25.0 * 1000.0),
    )
    optimizer.set_lr_scheduler(lr_scheduler)

    start = time.time()
    train_loss, val_loss = 100000.0, 100000.0
    prof = None
    train_loss_list, val_loss_list = [], []
    val_loader = DataLoader(val_dataset, batch_size=1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    if enable_test:
        config.checkpoint = Path(config.checkpoint).with_suffix(".pdparams")
        state_dict = paddle.load(config.checkpoint.as_posix())
        model.set_state_dict(state_dict)
        loss_press, loss_velo, p_orig, v_orig, spearmanr, loss_cd = test(
            model, val_loader, coef_norm, enable_test=True
        )
        log.info(
            f"val_loss = {(loss_press + reg*loss_velo):.4f}, Spearman's Rank Correlations = {spearmanr:.4f}, loss_velo = {v_orig.item():.4f}, loss_press = {p_orig.item():.4f}, loss_cd = {loss_cd:.4f}"
        )
        return

    pbar_train = tqdm(range(config.num_epochs), position=0)
    pbar_train.update(1)

    for epoch in range(config.num_epochs):
        std = time.time()
        loss_velo, loss_press = train_epoch(
            model, train_loader, optimizer, lr_scheduler, reg=reg, epoch=epoch, prof=prof
        )
        log.info(f"time cost: {(time.time() - std):4f}")
        train_loss = loss_velo + reg * loss_press
        if epoch == config.num_epochs - 1 or epoch % val_iter == 0:
            loss_press, loss_velo, p_orig, v_orig, spearmanr, loss_cd = test(
                model, val_loader, coef_norm, enable_test=enable_test
            )
            log.info(
                f"val_loss = {(loss_press + reg*loss_velo):.4f}, Sp R = {spearmanr:.4f}, loss_velo = {v_orig.item():.4f}, loss_press = {p_orig.item():.4f}, loss_cd = {loss_cd:.4f}"
            )
            val_loss = loss_velo + reg * loss_press
            save_checkpoint(path, epoch, model)
        if epoch != 0:
            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
            pbar_train.update(1)
        train_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())
    np.savetxt(f"{path}/train_loss_{config.num_epochs}.txt", train_loss_list)
    np.savetxt(f"{path}/val_loss_{config.num_epochs}.txt", val_loss_list)
    end = time.time()
    time_elapsed = end - start
    log.info("Number of parameters:", get_nb_trainable_params(model))
    log.info("Time elapsed: {0:.2f} seconds".format(time_elapsed))
    return model


@hydra.main(version_base=None, config_path="./configs", config_name="transolver_shapenetcar.yaml")
def main(config):
    paddle.device.set_device(f"gpu:{int(config.gpu)}")
    train_data, val_data, coef_norm = load_train_val_fold(config)
    train_ds = GraphDataset(train_data, use_cfd_mesh=False, r=0.2)
    val_ds = GraphDataset(val_data, use_cfd_mesh=False, r=0.2)
    if config.model_name == "Transolver":
        model = Model(
            n_hidden=256,
            n_layers=8,
            space_dim=7,
            fun_dim=0,
            n_head=8,
            mlp_ratio=2,
            out_dim=4,
            slice_num=32,
            unified_pos=False,
        )

        if config.enable_cinn:
            model = paddle.jit.to_static(
                model, full_graph=True, input_spec=[paddle.static.InputSpec(shape=[32186, 7], dtype="float32")]
            )
    else:
        raise NotImplementedError

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"output/{config.model_name}/{timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)

    train(
        config,
        train_ds,
        val_ds,
        model,
        path=path,
        val_iter=config.val_freq,
        coef_norm=coef_norm,
        enable_test=True if config.mode == "test" else False,
        enable_prof=config.enable_profiler,
    )


if __name__ == "__main__":
    main()
