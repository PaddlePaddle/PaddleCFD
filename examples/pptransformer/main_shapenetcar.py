import argparse
import os

import paddle
from ppcfd.data import GraphDataset
from ppcfd.data.shapenetcar_datamodule import load_train_val_fold
from ppcfd.networks.Transolver_orig import Model

import json
import os
import time
from typing import Tuple

import numpy as np
import paddle
from ppcfd.data.shapenetcar_datamodule import Data
from paddle.io import DataLoader
from tqdm import tqdm
from ppcfd.utils.profiler import init_profiler, update_profiler


def custom_collate_fn(batch: Tuple["Data", paddle.Tensor]):
    """自定义collate_fn，用于处理单个Data类型的数据项，直接返回单个数据和shape。"""
    data, shape = batch[0]

    # 提取 cfd_data 的各属性
    pos = data.pos
    x = data.x
    y = data.y
    surf = data.surf
    edge_index = data.edge_index

    # 创建新的 Data 对象
    single_data = Data(pos=pos, x=x, y=y, surf=surf, edge_index=edge_index)

    # 直接返回单个 Data 和 shape
    return single_data, shape


def get_nb_trainable_params(model):
    """
    Return the number of trainable parameters
    """
    model_parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    return sum([np.prod(tuple(p.shape)) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1, epoch=0, prof=None):
    model.train()
    criterion_func = paddle.nn.MSELoss(reduction="none")
    losses_press = []
    losses_velo = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.clear_gradients(set_to_zero=False)
        out = model((cfd_data, geom))
        targets = cfd_data.y
        loss_press = criterion_func(
            out[cfd_data.surf, -1], targets[cfd_data.surf, -1]
        ).mean(axis=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(axis=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        prof = update_profiler(True, prof, epoch)
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


@paddle.no_grad()
def test(device, model, test_loader):
    model.eval()
    criterion_func = paddle.nn.MSELoss(reduction="none")
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y
        loss_press = criterion_func(
            out[cfd_data.surf, -1], targets[cfd_data.surf, -1]
        ).mean(axis=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(axis=0)
        loss_velo = loss_velo_var.mean()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(
    device,
    train_dataset,
    val_dataset,
    Net,
    hparams,
    path,
    reg=1,
    val_iter=1,
    coef_norm=[],
):
    model = Net.to(device)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=hparams["lr"], weight_decay=0.0
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=(len(train_dataset) // hparams["batch_size"] + 1) * hparams["nb_epochs"],
        eta_min=hparams["lr"] / 1000,
        learning_rate=optimizer.get_lr(),
    )
    optimizer.set_lr_scheduler(tmp_lr)
    lr_scheduler = tmp_lr
    start = time.time()
    train_loss, val_loss = 100000.0, 100000.0
    pbar_train = tqdm(range(hparams["nb_epochs"]), position=0)
    prof = init_profiler(True)
    for epoch in pbar_train:
        train_loader = DataLoader(
            train_dataset,
            batch_size=hparams["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )
        loss_velo, loss_press = train(
            device, model, train_loader, optimizer, lr_scheduler, reg=reg, epoch=0, prof=None
        )
        train_loss = loss_velo + reg * loss_press
        del train_loader
        if val_iter is not None and (
            epoch == hparams["nb_epochs"] - 1 or epoch % val_iter == 0
        ):
            val_loader = DataLoader(
                val_dataset, batch_size=1, collate_fn=custom_collate_fn
            )
            loss_velo, loss_press = test(device, model, val_loader)
            val_loss = loss_velo + reg * loss_press
            del val_loader
            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss)
    end = time.time()
    time_elapsed = end - start
    params_model = float(get_nb_trainable_params(model))  # 确保 params_model 是浮点数
    print("Number of parameters:", params_model)
    print("Time elapsed: {0:.2f} seconds".format(time_elapsed))

    # 保存模型权重
    model_path = os.path.join(path, f"model_{hparams['nb_epochs']}.pdparams")
    paddle.save(model.state_dict(), model_path)

    # 记录日志
    if val_iter is not None:
        log_path = os.path.join(path, f"log_{hparams['nb_epochs']}.json")
        with open(log_path, "a") as f:
            log_data = {
                "nb_parameters": params_model,
                "time_elapsed": time_elapsed,
                "hparams": hparams,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "coef_norm": list(coef_norm),
            }
            json.dump(log_data, f, indent=4, cls=NumpyEncoder)

    return model

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/workspace/transolver/transolver_torch/Car-Design-ShapeNetCar/data/training_data")
parser.add_argument("--save_dir", default="/workspace/transolver/transolver_torch/Car-Design-ShapeNetCar/data/preprocessed_data")
parser.add_argument("--fold_id", default=0, type=int)
parser.add_argument("--gpu", default=3, type=int)
parser.add_argument("--val_iter", default=10, type=int)
parser.add_argument("--cfd_config_dir", default="cfd/cfd_params.yaml")
parser.add_argument("--cfd_model", default="Transolver")
parser.add_argument("--cfd_mesh", default=False)
parser.add_argument("--r", default=0.2, type=float)
parser.add_argument("--weight", default=0.5, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--nb_epochs", default=200, type=int)
parser.add_argument("--preprocessed", default=1, type=int)
args = parser.parse_args()
print(args)
hparams = {"lr": args.lr, "batch_size": args.batch_size, "nb_epochs": args.nb_epochs}
n_gpu = paddle.device.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and paddle.device.cuda.device_count() >= 1
device = str(f"cuda:{args.gpu}" if use_cuda else "cpu").replace("cuda", "gpu")
print(device)
train_data, val_data, coef_norm = load_train_val_fold(
    args, preprocessed=args.preprocessed
)
train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
if args.cfd_model == "Transolver":
    model = Model(
        n_hidden=256,
        n_layers=8,
        space_dim=7,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=32,
        unified_pos=0,
    ).to(device)
path = f"metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}"
if not os.path.exists(path):
    os.makedirs(path)
model = main(
    device,
    train_ds,
    val_ds,
    model,
    hparams,
    path,
    val_iter=args.val_iter,
    reg=args.weight,
    coef_norm=coef_norm,
)
