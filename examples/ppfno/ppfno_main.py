import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["HYDRA_FULL_ERROR"] = "0"

import paddle
from paddle.distributed import ParallelEnv, fleet
from paddle.io import DataLoader, DistributedBatchSampler

strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)

import logging
from timeit import default_timer
from typing import List, Union

import hydra
import meshio
import numpy as np
from omegaconf import DictConfig
from src.data import instantiate_datamodule
from src.losses import LpLoss
from src.networks import instantiate_network
from src.optim.schedulers import instantiate_scheduler
from src.utils.average_meter import AverageMeter, AverageMeterDict
from src.utils.dot_dict import DotDict, flatten_dict


def set_seed(seed: int = 0):
    paddle.seed(seed=seed)
    np.random.seed(seed)


def train(cfg: DictConfig):
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"/{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    model = instantiate_network(cfg)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-06)
    loss_fn = LpLoss(size_average=True)
    if cfg.enable_ddp:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    resume_ep = cfg.resume_ep
    if cfg.state:
        state = paddle.load(path=str(cfg.state))
        model.set_state_dict(state_dict=state["model"])
        optimizer.set_lr(state["lr"])
        resume_ep = state["epoch"]
        logging.info(f"Resuming model from epoch {resume_ep}.")

    device = ParallelEnv().device_id

    memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (1024 * 1024 * 1024)
    logging.info(f"Memory usage with model loading: {memory_allocated:.2f} GB")

    datamodule = instantiate_datamodule(cfg, cfg.n_train_num, 0, cfg.n_test_num)
    train_dataloader = datamodule.train_dataloader(enable_ddp=cfg.enable_ddp, batch_size=cfg.batch_size)

    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=cfg.num_epochs, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr

    logging.info(f"Start training {cfg.model} ...")

    for ep in range(cfg.num_epochs):
        if ep <= resume_ep:
            continue
        if ep == resume_ep + 1:
            logging.info(f"lr of {ep} is {optimizer.get_lr():.2e}")

        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeterDict()
        num_OOM = 0
        idx_batch = 0
        msg = "|| "
        for data_dict in train_dataloader:
            try:
                optimizer.clear_gradients(set_to_zero=False)
                pred, truth = model(data_dict)
                if idx_batch == 0 and paddle.distributed.get_rank() == 0:
                    msg = f"Data Loading Time: {data_dict['Data_loading_time'][0]:.2f} seconds. || "
                    memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (1024 * 1024 * 1024)
                    msg += f"Memory Usage: {memory_allocated:.2f} GB (forward), "
            except MemoryError as e:
                if "Out of memory" in str(e):
                    num_OOM += 1
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    continue
                else:
                    raise
            loss = paddle.to_tensor(data=0.0).cuda(blocking=True)
            for i in range(len(cfg.out_keys)):
                key = cfg.out_keys[i]
                st, end = sum(cfg.out_channels[:i]), sum(cfg.out_channels[:i]) + cfg.out_channels[i]
                loss_key = loss_fn(pred[st:end], truth[st:end])
                train_l2_meter.update({key: loss_key})
                loss += cfg.weight_list[i] * loss_key
            loss.backward(grad_tensor=loss)
            if idx_batch == 0 and paddle.distributed.get_rank() == 0:
                memory_allocated = paddle.device.cuda.memory_allocated(device=device) / 1024**3
                msg += f"{memory_allocated:.2f} GB (backward), "
                max_memory_allocated = paddle.device.cuda.max_memory_allocated(device=device) / 1024**3
                msg += f"{max_memory_allocated:.2f} GB (MAX), "
                memory_researved = paddle.device.cuda.memory_reserved() / 1024**3
                msg += f"{memory_researved:.2f} GB (Reserved)."
            optimizer.step()
            idx_batch += 1
        scheduler.step()
        t2 = default_timer()
        if num_OOM != 0:
            logging.info(f"WARNING: {num_OOM} samples OOM, skipping these samples.")
        msg_ep = f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2_Loss: "
        train_dict = train_l2_meter.avg
        for k, v in train_dict.items():
            msg_ep += f"{v.item():.4f}({k}), "
        if paddle.distributed.get_rank() == 0 and "msg" in locals():
            logging.info(msg_ep + msg)
        if ep == 0 or (ep + 1) % cfg.save_per_epoch == 0 or ep == cfg.num_epochs - 1:
            state = {"model": model.state_dict(), "lr": optimizer.get_lr(), "epoch": ep}
            paddle.save(obj=state, path=f"{cfg.output_dir}/{cfg.model}_{ep}.pdparams")


@paddle.no_grad()
def evaluate(cfg: DictConfig):
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"/{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    model = instantiate_network(cfg)
    loss_fn = LpLoss(size_average=True)
    if cfg.enable_ddp:
        model = fleet.distributed_model(model)
    assert cfg.state is not None, "checkpoint must be given."
    state = paddle.load(path=str(cfg.state))
    model.set_state_dict(state_dict=state["model"])

    device = ParallelEnv().device_id
    memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (1024 * 1024 * 1024)
    logging.info(f"Memory usage with model loading: {memory_allocated:.2f} GB")

    datamodule = instantiate_datamodule(cfg, cfg.n_train_num, 0, cfg.n_test_num)
    test_dataloader = datamodule.test_dataloader(enable_ddp=cfg.enable_ddp, batch_size=cfg.batch_size)
    all_files = os.listdir(os.path.join(cfg.data_path, cfg.mode))
    prefix = "area"
    indices = [item[5:9] for item in all_files if item.startswith(prefix)]

    def extract_number(s):
        return int(s)

    indices.sort(key=extract_number)
    logging.info(f"Start evaluting {cfg.model} ...")
    t1 = default_timer()
    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.eval()
    eval_meter = AverageMeterDict()

    def cal_mre(pred, label):
        return paddle.abs(x=pred - label) / paddle.abs(x=label)

    for i, data_dict in enumerate(test_dataloader):
        device = ParallelEnv().device_id
        device = paddle.CUDAPlace(device)
        try:
            out_dict, pred, truth = model.eval_dict(device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode)
            if cfg.save_eval_results:
                save_eval_results(cfg, pred, truth, indices[i], decode_fn=datamodule.decode)
            paddle.device.cuda.empty_cache()
        except MemoryError as e:
            if "Out of memory" in str(e):
                logging.info(f"WARNING: OOM on sample {i}, skipping this sample.")
                if hasattr(paddle.device.cuda, "empty_cache"):
                    paddle.device.cuda.empty_cache()
                continue
            else:
                raise
        msg = f"Eval sample {i}... L2_Error: "
        for k, v in out_dict.items():
            if k.split("_")[0] == "L2":
                msg += f"{k}: {v.item():.4f}, "
                eval_meter.update({k: v})
        msg += "|| MRE and Value: "
        for k, v in out_dict.items():
            if "Cd" and "pred" in k.split("_"):
                k_truth = f"{k[:k.rfind('_')]}_truth"
                mre = cal_mre(v, out_dict[k_truth])
                eval_meter.update({f"MRE_{k[:k.rfind('_')]}": mre})
                msg += f"MRE_{k[:k.rfind('_')]}: {mre.item():.4f}, "
                msg += f"[{k}: {v:.4f}, {k_truth}: {out_dict[k_truth]:.4f}], "
        logging.info(msg)
    t2 = default_timer()
    msg = f"Testing took {t2 - t1:.2f} seconds. Everage eval values: "
    eval_dict = eval_meter.avg
    for k, v in eval_dict.items():
        msg += f"{v.item():.4f}({k}), "
    logging.info(msg)
    max_memory_allocated = paddle.device.cuda.max_memory_allocated(device=device) / (1024 * 1024 * 1024)
    print(f"Memory Usage: {max_memory_allocated:.2f} GB (MAX).")


def save_eval_results(cfg: DictConfig, pred, truth, centroid_idx, decode_fn=None):
    pred_pressure = decode_fn(pred[0:1, :], 0).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1).cpu().detach().numpy()
    truth_pressure = decode_fn(truth[0:1, :], 0).cpu().detach().numpy()
    truth_wallshearstress = decode_fn(truth[1:4, :], 1).cpu().detach().numpy()
    delta_pressure = pred_pressure - truth_pressure
    delta_wallshearstress = pred_wallshearstress - truth_wallshearstress
    evals_results = {
        "pred_pressure": pred_pressure,
        "pred_wallshearstress": pred_wallshearstress,
        "truth_pressure": truth_pressure,
        "truth_wallshearstress": truth_wallshearstress,
        "delta_pressure": delta_pressure,
        "delta_wallshearstress": delta_wallshearstress,
    }
    centroid = np.load(f"{cfg.data_path}/{cfg.mode}/centroid_{centroid_idx}.npy")
    cells = [("vertex", np.arange(tuple(centroid.shape)[0]).reshape(-1, 1))]
    mesh = meshio.Mesh(points=centroid, cells=cells)
    for k, v in evals_results.items():
        np.save(os.path.join(f"{cfg.data_path}/evals_results", f"{k}_{centroid_idx}.npy"), v.T)
        mesh.point_data.update({f"{k}": v.T})
    mesh.write(f"{cfg.data_path}/evals_results/eval_{centroid_idx}.vtk")
    return None


@hydra.main(config_path="./configs", config_name="gi_fno")
def main(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)
    if cfg.mode == "train":
        print("################## training #####################")
        train(cfg)
    elif cfg.mode == "test":
        print("################## test #####################")
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'test'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
