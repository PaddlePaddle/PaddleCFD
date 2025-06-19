import argparse
import os
from re import M

import paddle
from pathlib import Path
import scipy
from sklearn.isotonic import spearmanr
# paddle.set_device('cpu')
from ppcfd.data import GraphDataset
from ppcfd.data.shapenetcar_datamodule import load_train_val_fold
from ppcfd.networks.Transolver_orig import Model
from datetime import datetime
from drag_coefficient import cal_coefficient

import json
import os
import time

import numpy as np
import paddle
from paddle.io import DataLoader
from tqdm import tqdm
from ppcfd.utils.profiler import init_profiler, update_profiler
import meshio
paddle.seed(42)
np.random.seed(42)

def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    return sum([np.prod(tuple(p.shape)) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, reg=1, epoch=0, prof=None):
    model.train()
    criterion_func = paddle.nn.MSELoss(reduction="none")
    losses_press = []
    losses_velo = []
    for data in train_loader:
        inputs, targets, surf, sample_name = data
        B = inputs.shape[0]
        optimizer.clear_grad()
        out = model(inputs[0])
        p_pred = paddle.stack([out[i][surf[i], -1:] for i in range(B)], axis=0)
        p_true = paddle.stack([targets[i][surf[i], -1:] for i in range(B)], axis=0)
        loss_press = criterion_func(p_pred, p_true).mean()
        loss_velo_var = criterion_func(out[:, :, :-1], targets[:, :, :-1]).mean()
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        prof = update_profiler(False, prof, epoch)
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
        losses_press.append(total_loss.item())
        losses_velo.append(total_loss.item())
    return np.mean(losses_press), np.mean(losses_velo)


@paddle.no_grad()
def test(device, model, test_loader, coef_norm, enable_test=False):
    model.eval()
    criterion_func = paddle.nn.MSELoss(reduction="none")
    losses_press, losses_velo, losses_cd, p_orig, v_orig = [], [], [], [], []
    cd_pred_list, cd_true_list = [], []
    for i, data in enumerate(test_loader):
        inputs, targets, surf, sample_name = data
        B = inputs.shape[0]
        out = model(inputs[0])
        p_pred = paddle.stack([out[j][surf[j], -1:] for j in range(B)], axis=0)
        p_true = paddle.stack([targets[j][surf[j], -1:] for j in range(B)], axis=0)

        eps = 0.
        # inputs_orig = (inputs * (coef_norm[1] + eps) + coef_norm[0])
        out_orig = (out * (coef_norm[3] + eps) + coef_norm[2])
        targets_orig = (targets *(coef_norm[3] + eps) + coef_norm[2])
        p_pred_orig = paddle.stack([out_orig[j][surf[j], -1:] for j in range(B)], axis=0)
        p_true_orig = paddle.stack([targets_orig[j][surf[j], -1:] for j in range(B)], axis=0)

        v_pred_orig = paddle.stack([out_orig[j][:, :-1] for j in range(B)], axis=0)
        v_true_orig = paddle.stack([targets_orig[j][:, :-1] for j in range(B)], axis=0)

        loss_press_orig = paddle.linalg.norm(p_pred_orig - p_true_orig) / paddle.linalg.norm(p_true_orig)
        loss_velo_orig = paddle.linalg.norm(v_true_orig - v_pred_orig) / paddle.linalg.norm(v_pred_orig)
        # import pdb;pdb.set_trace()

        # cd_pred = cal_coefficient(Path(args.data_dir) / Path(sample_name[0]), p_pred_orig[0], v_pred_orig[0])
        # cd_true = cal_coefficient(Path(args.data_dir) / Path(sample_name[0]), p_true_orig[0], v_true_orig[0])
        # cd_pred_list.append(cd_pred)
        # cd_true_list.append(cd_true)
        # loss_cd = abs(cd_pred - cd_true) / abs(cd_true)

        if enable_test == True:
            print(f"{i} {sample_name[0]}, loss_velo = {loss_velo_orig.item():.4f}, loss_press = {loss_press_orig.item():.4f}")
            # print(f"{i} {sample_name[0]}, loss_velo = {loss_velo_orig.item():.4f}, loss_press = {loss_press_orig.item():.4f}, loss_cd = {loss_cd.item():.4f}")
        # centroid = paddle.stack([inputs_orig[j][surf[j], :3] for j in range(B)], axis=0)
        # for j in range(B):
        #     cells = [('vertex', np.arange(tuple(centroid[j].shape)[0]).reshape(-1, 1))]
        #     mesh = meshio.Mesh(
        #         points=centroid[j].numpy(),
        #         point_data={
        #             'p_pred': p_pred_orig[j].numpy(),
        #             'p_true': p_true_orig[j].numpy(),
        #             'error': abs((p_pred_orig[j]-p_true_orig[j])).numpy()
        #             },
        #         cells=cells
        #         )
        #     print(f'save file : ./metrics/eval_[{sample_name[j]}].vtk')
        #     mesh.write(f'./metrics/eval_{j}.vtk')
        loss_press = criterion_func(p_pred, p_true).mean()
        loss_velo_var = criterion_func(out[:, :, :-1], targets[:, :, :-1]).mean()
        loss_velo = loss_velo_var.mean()
        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
        p_orig.append(loss_press_orig)
        v_orig.append(loss_velo_orig)

        # losses_cd.append(loss_cd)

    cd_pred_list = np.array(cd_pred_list)
    cd_true_list = np.array(cd_true_list)
    # spear = scipy.stats.spearmanr(cd_true_list, cd_pred_list)[0]
    spear = 0.0
    # cd_mre = np.mean(losses_cd)
    cd_mre = 0.0
    return np.mean(losses_press), np.mean(losses_velo), np.mean(p_orig), np.mean(v_orig), spear, cd_mre


def save_checkpoint(path, epoch, model):
    model_path = os.path.join(path, f"model_{epoch+1}.pdparams")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    paddle.save(state_dict, model_path)


def main(
    device,
    train_dataset,
    val_dataset,
    model,
    hparams,
    path,
    reg=1,
    val_iter=1,
    coef_norm=[],
    enable_test=False
):
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=hparams["lr"], weight_decay=0.0
    )
    tmp_lr = paddle.optimizer.lr.OneCycleLR(
        max_learning_rate=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        end_learning_rate=hparams['lr']/(25.*1000.))
    optimizer.set_lr_scheduler(tmp_lr)
    lr_scheduler = tmp_lr

    start = time.time()
    train_loss, val_loss = 100000.0, 100000.0
    prof = init_profiler(False)
    print("Start training...")
    custom_collate_fn = None
    train_loss_list, val_loss_list = [], []
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    if enable_test:
        print("testing...")
        state_dict = paddle.load("./metrics/Transolver/20250619_091023/model_131.pdparams")
        model.set_state_dict(state_dict)
        loss_press, loss_velo, p_orig, v_orig, spearmanr, loss_cd = test(device, model, val_loader, coef_norm, enable_test=True)
        print(f"val_loss = {(loss_press + reg*loss_velo):.4f}, Spearman's Rank Correlations = {spearmanr:.4f}, loss_velo = {v_orig.item():.4f}, loss_press = {p_orig.item():.4f}, loss_cd = {loss_cd:.4f}")
        return

    pbar_train = tqdm(range(hparams["nb_epochs"]), position=0)
    for epoch in pbar_train:
        loss_velo, loss_press = train(
            device, model, train_loader, optimizer, lr_scheduler, reg=reg, epoch=epoch, prof=prof
        )
        train_loss = loss_velo + reg * loss_press
        if epoch == hparams["nb_epochs"] - 1 or epoch % val_iter == 0:
            loss_press, loss_velo, p_orig, v_orig, spearmanr, loss_cd = test(device, model, val_loader, coef_norm, enable_test=True)
            print(f"val_loss = {(loss_press + reg*loss_velo):.4f}, Spearman's Rank Correlations = {spearmanr:.4f}, loss_velo = {v_orig.item():.4f}, loss_press = {p_orig.item():.4f}, loss_cd = {loss_cd:.4f}")
            val_loss = loss_velo + reg * loss_press
            save_checkpoint(path, epoch, model)
        pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        train_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())
    np.savetxt(f"{path}/train_loss_{hparams['nb_epochs']}.txt", train_loss_list)
    np.savetxt(f"{path}/val_loss_{hparams['nb_epochs']}.txt", val_loss_list)
    end = time.time()
    time_elapsed = end - start
    params_model = float(get_nb_trainable_params(model))
    print("Number of parameters:", params_model)
    print("Time elapsed: {0:.2f} seconds".format(time_elapsed))
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/workspace/transolver/transolver_torch/Car-Design-ShapeNetCar/data/training_data")
parser.add_argument("--save_dir", default="/workspace/transolver/transolver_torch/Car-Design-ShapeNetCar/data/preprocessed_data")
parser.add_argument("--fold_id", default=0, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--val_iter", default=10, type=int)
parser.add_argument("--cfd_config_dir", default="cfd/cfd_params.yaml")
parser.add_argument("--cfd_model", default="Transolver")
parser.add_argument("--cfd_mesh", default=False)
parser.add_argument("--enable_test", default=False)
parser.add_argument("--r", default=0.2, type=float)
parser.add_argument("--weight", default=0.5, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--nb_epochs", default=155, type=int)
parser.add_argument("--preprocessed", default=1, type=int)
parser.add_argument("--cinn", default=False, type=bool)
parser.add_argument("--n_train", default=800, type=int)
parser.add_argument("--n_eval", default=200, type=int)
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
        unified_pos=False,
    )
    print("args.cinn", args.cinn)
    state_dict = {k: paddle.to_tensor(v).astype(paddle.float32) for k, v in state_dict["model_state"].items()}
    for k in state_dict.keys():
        if "weight" in k and "ln" not in k:
            # print(k, state_dict[k].shape)
            state_dict[k] = state_dict[k].transpose([1, 0])
    model.set_state_dict(state_dict)

    if args.cinn:
        import einops
        paddle.jit.ignore_module([einops])
        paddle.framework.core._set_prim_all_enabled(True)
        model = paddle.jit.to_static(model, full_graph=True, backend='CINN', input_spec = [paddle.static.InputSpec(shape=[1, None, 7], dtype='float32')])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"metrics/{args.cfd_model}/{timestamp}"
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
    enable_test=args.enable_test
)
