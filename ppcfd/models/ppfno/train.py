from __future__ import annotations

import json
import logging
import os
import sys


sys.path.append("./src")
sys.path.append("./src/networks")
import random
from timeit import default_timer
from typing import Dict
from typing import Tuple

import hydra
import numpy as np
import paddle
import pyvista as pv
from omegaconf import DictConfig
from paddle import distributed as dist
from paddle.distributed import ParallelEnv
from paddle.distributed import fleet
from src.data import instantiate_datamodule
from src.losses import LpLoss
from src.networks import instantiate_network
from src.utils.average_meter import AverageMeterDict


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["HYDRA_FULL_ERROR"] = "0"


def set_seed(seed: int = 0):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


world_size = dist.get_world_size()
if world_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.find_unused_parameters = True
    fleet.init(is_collective=True, strategy=strategy)

print(f"total gpu num: {world_size}")


def save_vtp_from_dict(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
    num_timestamps: int = 1,
):
    """Save dict data to '*.vtp' file.

    Args:
        filename (str): Output filename.
        data_dict (Dict[str, np.ndarray]): Data in dict.
        coord_keys (Tuple[str, ...]): Tuple of coord key. such as ("x", "y").
        value_keys (Tuple[str, ...]): Tuple of value key. such as ("u", "v").
        num_timestamps (int, optional): Number of timestamp in data_dict. Defaults to 1.

    Examples:
        >>> import ppsci
        >>> import numpy as np
        >>> filename = "path/to/file.vtp"
        >>> data_dict = {
        ...     "x": np.array([[1], [2], [3],[4]]),
        ...     "y": np.array([[2], [3], [4],[4]]),
        ...     "z": np.array([[3], [4], [5],[4]]),
        ...     "u": np.array([[4], [5], [6],[4]]),
        ...     "v": np.array([[5], [6], [7],[4]]),
        ... }
        >>> coord_keys = ("x","y","z")
        >>> value_keys = ("u","v")
        >>> ppsci.visualize.save_vtp_from_dict(filename, data_dict, coord_keys, value_keys) # doctest: +SKIP
    """

    if len(coord_keys) not in [3]:
        raise ValueError(f"ndim of coord ({len(coord_keys)}) should be 3 in vtp format")

    coord = [data_dict[k] for k in coord_keys if k not in ("t", "sdf")]
    assert all([c.ndim == 2 for c in coord]), "array of each axis should be [*, 1]"
    coord = np.concatenate(coord, axis=1)

    if not isinstance(coord, np.ndarray):
        raise ValueError(f"type of coord({type(coord)}) should be ndarray.")
    if len(coord) % num_timestamps != 0:
        raise ValueError(
            f"coord length({len(coord)}) should be an integer multiple of " f"num_timestamps({num_timestamps})"
        )
    if coord.shape[1] not in [3]:
        raise ValueError(f"ndim of coord({coord.shape[1]}) should be 3 in vtp format.")

    if len(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    npoint = len(coord)
    nx = npoint // num_timestamps
    if filename.endswith(".vtp"):
        filename = filename[:-4]

    for t in range(num_timestamps):
        coord_ = coord[t * nx : (t + 1) * nx]
        point_cloud = pv.PolyData(coord_)
        for k in value_keys:
            value_ = data_dict[k][t * nx : (t + 1) * nx]
            if value_ is not None and not isinstance(value_, np.ndarray):
                raise ValueError(f"type of value({type(value_)}) should be ndarray.")
            if value_ is not None and len(coord_) != len(value_):
                raise ValueError(f"coord length({len(coord_)}) should be equal to value length({len(value_)})")
            point_cloud[k] = value_

        if num_timestamps > 1:
            width = len(str(num_timestamps - 1))
            point_cloud.save(f"{filename}_t-{t:0{width}}.vtp")
        else:
            point_cloud.save(f"{filename}.vtp")

    if num_timestamps > 1:
        logging.info(
            f"Visualization results are saved to: {filename}_t-{0:0{width}}.vtp ~ "
            f"{filename}_t-{num_timestamps - 1:0{width}}.vtp"
        )
    else:
        logging.info(f"Visualization result is saved to: {filename}.vtp")


def train(cfg: DictConfig):
    os.makedirs(cfg.train_output_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.train_output_path, "log"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg.train_output_path, "log", f"{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
        force=True,
    )

    # 创建流处理器（终端输出）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(stream_handler)

    # train_json_file_path = os.path.join('/home/chenkai26/Paddle-AeroSimOpt/output/dataset1/json', "train.json")
    # coefficent_json_file_path = os.path.join('/home/chenkai26/Paddle-AeroSimOpt/output/dataset1//json', 'coefficent.json')
    # /home/chenkai26/Paddle-AeroSimOpt/refine_data/dataset1
    # output_dir = cfg.train_input_path.replace("refine_data", "output")
    os.makedirs(os.path.join(cfg.train_output_path, "json"), exist_ok=True)
    train_json_file_path = os.path.join(cfg.train_output_path, "json", "train.json")
    coefficent_json_file_path = os.path.join(cfg.train_output_path, "json", "coefficent.json")

    def create_json(json_file_path):
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        if os.path.isfile(json_file_path):
            os.remove(json_file_path)
        with open(json_file_path, "w") as file:
            json.dump([], file)

    create_json(train_json_file_path)
    create_json(coefficent_json_file_path)

    def append_dict_to_json_list(file_path, dict_element):
        assert os.path.exists(file_path), file_path
        with open(file_path, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            data.append(dict_element)
        else:
            logging.info("Error: The root of the JSON file is not a list.")
            return
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

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

    datamodule = instantiate_datamodule(
        cfg,
        cfg.train_input_path,
        cfg.n_train_num,
        0,
        cfg.n_test_num,
        cfg.train_ratio,
        cfg.test_ratio,
    )
    train_dataloader = datamodule.train_dataloader(enable_ddp=cfg.enable_ddp, batch_size=cfg.batch_size)

    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=cfg.num_epochs, learning_rate=optimizer.get_lr())
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr

    logging.info(f"Start training {cfg.model} ...")

    # evaluate init
    test_dataloader = datamodule.test_dataloader(
        enable_ddp=False, batch_size=cfg.batch_size
    )  # each GPU use full test dataset
    # all_files = os.listdir(cfg.train_input_path)
    # prefix = "area"
    os.makedirs(os.path.join(cfg.train_output_path, "json"), exist_ok=True)
    """
    {
        "test_case_id":
        [
        SFE-CR450AF-U3-FZ-001-202503,
        SFE-CR450AF-U3-FZ-001-202504
        ],
        "train_case_id":
        [
        SFE-CR450AF-U3-FZ-001-202505,
        SFE-CR450AF-U3-FZ-001-202506
        ]
    }
    """
    data = {
        "test_case_id": datamodule.test_full_caseids,
        "train_case_id": datamodule.train_full_caseids,
    }
    if paddle.distributed.get_rank() == 0:
        with open(os.path.join(cfg.train_output_path, "json", "radius.json"), "w") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
    # indices = [item[5:9] for item in all_files if item.startswith(prefix) and item.endswith(".npy")]

    # def extract_number(s):
    #     return int(s)

    # indices.sort(key=extract_number)
    # if isinstance(model, paddle.DataParallel):
    #     eval_model = model._layers
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []

    if paddle.distributed.get_rank() == 0:
        logging.info(f"train indices: {datamodule.train_full_caseids}")
        logging.info(f"test indices: {datamodule.test_full_caseids}")

    def cal_mre(pred, label):
        return paddle.abs(x=pred - label) / paddle.abs(x=label)

    def evaluate_on_fly(epoch_id) -> int | None:
        t1 = default_timer()
        max_cd_error = 0.0
        max_loss_case_id = None
        coefficent_json_dict = []
        if paddle.distributed.get_rank() == 0:
            logging.info(f"Start evaluting {cfg.model} at epoch {epoch_id}, number of samples: {len(test_dataloader)}")

        # all_files = os.listdir(os.path.join(cfg.train_input_path, "test"))
        # all_files = os.listdir(cfg.train_input_path)
        # prefix = "area"
        # indices = [item[5:9] for item in all_files if item.startswith(prefix) and item.endswith(".npy")]
        indices = datamodule.test_indices
        full_indices = datamodule.test_full_caseids

        # def extract_number(s):
        #     return int(s)

        # indices.sort(key=extract_number)

        current_model = eval_model if "eval_model" in locals() else model
        is_train = current_model.training
        if is_train:
            current_model.eval()
        # for dataParallel
        if isinstance(current_model, paddle.DataParallel):
            current_model = current_model._layers
        for i, data_dict in enumerate(test_dataloader):
            case_coefficent_json_dict = {}
            device = ParallelEnv().device_id
            device = paddle.CUDAPlace(device)
            try:
                out_dict, pred, truth, cd_dict = current_model.eval_dict(
                    device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
                )

                if paddle.any(paddle.isnan(cd_dict["Cd_truth"])):
                    logging.info(f"WARNING: nan detected on test sample {i}, skipping this sample.")
                    continue

                if cfg.save_eval_results:
                    save_eval_results(
                        cfg,
                        pred,
                        truth,
                        indices[i],
                        epoch_id,
                        decode_fn=datamodule.decode,
                        caseid=datamodule.test_full_caseids[i],
                    )
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
                    if k == "Cd_pred" and max_cd_error < mre.item():
                        max_cd_error = mre.item()
                        max_loss_case_id = i

            Cd_pred_modify = cd_dict["Cd_pred_modify"]
            Cd_truth = out_dict["Cd_truth"]
            Cd_pred = out_dict["Cd_pred"]
            Cd_mre_modify = paddle.abs(x=Cd_pred_modify - Cd_truth) / paddle.abs(x=Cd_truth)
            case_coefficent_json_dict["cal_total_drag_coefficient"] = Cd_pred_modify.item()
            case_coefficent_json_dict["real_total_drag_coefficient"] = Cd_truth.item()
            case_coefficent_json_dict["cal_pressure_drag_coefficient"] = out_dict["Cd_pressure_pred"].item()
            case_coefficent_json_dict["real_pressure_drag_coefficient"] = out_dict["Cd_pressure_truth"].item()
            case_coefficent_json_dict["cal_friction_resistance_coefficient"] = out_dict[
                "Cd_wallshearstress_pred"
            ].item()
            case_coefficent_json_dict["real_friction_resistance_coefficient"] = out_dict[
                "Cd_wallshearstress_truth"
            ].item()
            case_coefficent_json_dict["cal_error_total_drag_coefficient"] = Cd_truth.item() - Cd_pred_modify.item()
            case_coefficent_json_dict["cal_err_pressure_drag_coefficient"] = (
                out_dict["Cd_pressure_truth"].item() - out_dict["Cd_pressure_pred"].item()
            )
            case_coefficent_json_dict["cal_err_friction_resistance_coefficient"] = (
                out_dict["Cd_wallshearstress_truth"].item() - out_dict["Cd_wallshearstress_pred"].item()
            )

            case_coefficent_json_dict["case_id"] = full_indices[i]
            # coefficent_json_dict.update(case_coefficent_json_dict)
            coefficent_json_dict.append(case_coefficent_json_dict)

            # eval_meter.update({"Cd_mre_modify": Cd_mre_modify})
            # eval_meter.update({"Cd_pred_modify": Cd_pred_modify})

            msg += f"MRE_Cd_modify: {Cd_mre_modify.item():.4f}, "
            msg += f"[Cd_pred_modify: {Cd_pred_modify.item():.4f}, "
            msg += f"Cd_truth: {Cd_truth.item():.4f}], "

            logging.info(msg)

        t2 = default_timer()
        msg = f"Testing took {t2 - t1:.2f} seconds. Everage eval values: "
        eval_dict = eval_meter.avg
        for k, v in eval_dict.items():
            msg += f"{v.item():.4f}({k}), "

        if is_train:
            current_model.train()

        if max_loss_case_id is not None:
            msg += f"Maximum Cd Error Sample ID: {datamodule.test_full_caseids[max_loss_case_id]}(index={max_loss_case_id}), Maximum Cd Error: {max_cd_error:.4f}"
        else:
            msg += (
                "Wawrning: No maximum Cd Error, because all samples are not evaluated, might for OMM or other reason."
            )
        logging.info(msg)
        # max_memory_allocated = paddle.device.cuda.max_memory_allocated(device=device) / (
        #     1024 * 1024 * 1024
        # )
        # print(f"Memory Usage: {max_memory_allocated:.2f} GB (MAX).")
        if max_loss_case_id is not None:
            return datamodule.test_full_caseids[max_loss_case_id], coefficent_json_dict
        else:
            return None, coefficent_json_dict

    for ep in range(cfg.num_epochs):
        if paddle.distributed.get_rank() == 0:
            train_json_dict = {}
        coefficent_json_dict = None
        if ep <= resume_ep:
            continue
        if ep == resume_ep + 1:
            logging.info(f"lr of {ep} is {optimizer.get_lr():.2e}")

        t1 = default_timer()
        train_l2_meter = AverageMeterDict()
        num_OOM = 0
        idx_batch = 0
        msg = "|| "

        if ep == cfg.num_epochs - cfg.finetuning_epochs + 1:
            tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=cfg.finetuning_epochs, learning_rate=cfg.lr_cd)
            optimizer.set_lr_scheduler(tmp_lr)
            scheduler = tmp_lr

        if ep <= cfg.num_epochs - cfg.finetuning_epochs:
            for name, param in model.named_parameters():
                if "integral_cd" not in name:
                    param.stop_gradient = False
                else:
                    param.stop_gradient = True
            msg = "Integral CD params are frozen. || "
            model.train()
        else:
            for name, param in model.named_parameters():
                if "integral_cd" in name:
                    param.stop_gradient = False
                else:
                    param.stop_gradient = True
            msg = "Other params are frozen. || "
            msg += f"lr_cd: {optimizer.get_lr():.2e}, "
            model.eval()

        for data_dict in train_dataloader:
            try:
                if idx_batch == 0 and paddle.distributed.get_rank() == 0:
                    msg += f"Data Loading Time: {data_dict['Data_loading_time'][0]:.2f} seconds. || "
                    memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (1024 * 1024 * 1024)
                    msg += f"Memory Usage: {memory_allocated:.2f} GB (forward), "

                optimizer.clear_gradients(set_to_zero=False)
                pred, truth, cd_dict = model(data_dict, idx_batch, loss_fn=loss_fn, decode_fn=datamodule.decode)
                if "OOM" in cd_dict:
                    if cd_dict["OOM"] == True:
                        idx_batch += 1
                        continue
                    elif cd_dict["OOM"] == False and paddle.any(paddle.isnan(cd_dict["Cd_truth"])):
                        logging.info(f"WARNING: nan detected on sample {idx_batch}, skipping this sample.")
                        idx_batch += 1
                        continue

            except MemoryError as e:
                if "Out of memory" in str(e):
                    num_OOM += 1
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    continue
                else:
                    raise
            loss = paddle.to_tensor(data=0.0).cuda(blocking=True)
            # print('cd_dict:', cd_dict)
            if cd_dict == {}:
                for i in range(len(cfg.out_keys)):
                    key = cfg.out_keys[i]
                    st, end = (
                        sum(cfg.out_channels[:i]),
                        sum(cfg.out_channels[:i]) + cfg.out_channels[i],
                    )
                    loss_key = loss_fn(pred[st:end], truth[st:end])
                    train_l2_meter.update({key: loss_key})
                    loss += cfg.weight_list[i] * loss_key
            else:
                Cd_pred_modify = cd_dict["Cd_pred_modify"]
                Cd_truth = cd_dict["Cd_truth"]
                Cd_pred = cd_dict["Cd_pred"]
                Cd_mre = paddle.abs(x=Cd_pred_modify - Cd_truth) / paddle.abs(x=Cd_truth)
                loss += paddle.nn.functional.mse_loss(Cd_pred_modify, Cd_truth)
                train_l2_meter.update({"pressure": cd_dict["L2_pressure"]})
                train_l2_meter.update({"wallshearstress": cd_dict["L2_wallshearstress"]})
                train_l2_meter.update({"MSE_loss": loss})
                train_l2_meter.update({"Cd_mre": Cd_mre})
                train_l2_meter.update({"Cd_pred": Cd_pred})
                train_l2_meter.update({"Cd_pred_modify": Cd_pred_modify})
                train_l2_meter.update({"Cd_truth": Cd_truth})

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

        if paddle.distributed.get_rank() == 0:
            train_json_dict["epoch"] = ep
            if "Cd_mre" in train_l2_meter.avg:
                train_json_dict["mre"] = train_l2_meter.avg["Cd_mre"].item()
            else:
                train_json_dict["mre"] = 0
            train_json_dict["pressure_loss"] = train_l2_meter.avg["pressure"].item()
            train_json_dict["shear_stress_loss"] = train_l2_meter.avg["wallshearstress"].item()

        if num_OOM != 0:
            logging.info(f"WARNING: {num_OOM} samples OOM, skipping these samples.")
        msg_ep = f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2_Loss: "
        train_dict = train_l2_meter.avg
        for k, v in train_dict.items():
            msg_ep += f"{v.item():.4f}({k}), "
        if paddle.distributed.get_rank() == 0 and "msg" in locals():
            logging.info(msg_ep + msg)
        max_loss_case_id = None
        if ep == 0 or (ep + 1) % cfg.save_per_epoch == 0 or ep == cfg.num_epochs - 1:
            # if (ep + 1) % cfg.save_per_epoch == 0 or ep == cfg.num_epochs - 1:
            state = {"model": model.state_dict(), "lr": optimizer.get_lr(), "epoch": ep}
            os.makedirs(
                os.path.dirname(f"{cfg.train_output_path}/pd/{cfg.model_name}.pdparams"),
                exist_ok=True,
            )
            paddle.save(obj=state, path=f"{cfg.train_output_path}/pd/{cfg.model_name}.pdparams")
            logging.info(f"Save checkpoint to: {cfg.train_output_path}/pd/{cfg.model_name}.pdparams")
            max_loss_case_id, coefficent_json_dict = evaluate_on_fly(ep)

        if paddle.distributed.get_rank() == 0:
            train_json_dict["max_loss_case_id"] = max_loss_case_id
            append_dict_to_json_list(train_json_file_path, train_json_dict)

        if paddle.distributed.get_rank() == 0:
            if isinstance(coefficent_json_dict, dict):
                append_dict_to_json_list(coefficent_json_file_path, coefficent_json_dict)
            if isinstance(coefficent_json_dict, list):
                for coefficent_json_dict_ in coefficent_json_dict:
                    append_dict_to_json_list(coefficent_json_file_path, coefficent_json_dict_)


def save_eval_results(
    cfg: DictConfig,
    pred,
    truth,
    centroid_idx,
    epoch_id,
    decode_fn=None,
    caseid=None,
):
    pred_pressure = decode_fn(pred[0:1, :], 0).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1).cpu().detach().numpy()
    truth_pressure = decode_fn(truth[0:1, :], 0).cpu().detach().numpy()
    truth_wallshearstress = decode_fn(truth[1:4, :], 1).cpu().detach().numpy()
    delta_pressure = pred_pressure - truth_pressure
    delta_wallshearstress = pred_wallshearstress - truth_wallshearstress
    evals_results = {
        "cal_pressure_drag": pred_pressure,
        "cal_friction_resistance": pred_wallshearstress,
        "real_pressure_drag": truth_pressure,
        "real_friction_resistance": truth_wallshearstress,
        "cal_err_pressure": delta_pressure,
        "cal_err_friction_resistance": delta_wallshearstress,
    }
    centroid = np.load(f"{cfg.train_input_path}/centroid_{centroid_idx}.npy")
    cells = [("vertex", np.arange(tuple(centroid.shape)[0]).reshape(-1, 1))]

    output_dir = cfg.train_output_path
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    for k, v in evals_results.items():
        # save 6 csv output files
        array_hstack = np.hstack((centroid, v.T))
        csv_filename = os.path.join(
            output_dir,
            "csv",
            str(epoch_id),
            str(caseid),
            f"{k}.csv",
        )
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        np.savetxt(csv_filename, array_hstack, delimiter=",", fmt="%f")
        logging.info(f"Save csv to: {csv_filename}")

        # save 6 vtp output files
        # mesh = meshio.Mesh(points=centroid, cells=cells)
        # mesh.point_data.update({f"{k}": v.T})
        vtp_filename = os.path.join(
            output_dir,
            "vtp",
            str(epoch_id),
            str(caseid),
            f"{k}.vtp",
        )
        os.makedirs(os.path.dirname(vtp_filename), exist_ok=True)
        # mesh.write(vtp_filename, file_format="vtk", binary=False)
        if v.T.shape[1] == 1:
            save_vtp_from_dict(
                vtp_filename,
                {
                    "x": centroid[:, 0:1],
                    "y": centroid[:, 1:2],
                    "z": centroid[:, 2:3],
                    k: v.T,
                },
                ("x", "y", "z"),
                (k,),
            )
        else:
            save_vtp_from_dict(
                vtp_filename,
                {
                    "x": centroid[:, 0:1],
                    "y": centroid[:, 1:2],
                    "z": centroid[:, 2:3],
                    k: np.linalg.norm(v.T, axis=1, keepdims=True),
                },
                ("x", "y", "z"),
                (k,),
            )

        logging.info(f"Save vtp to: {vtp_filename}")

    return None


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    cfg.enable_ddp = world_size > 1
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # dataset_merge()
    # dataset_split(train_percent, val_percent)

    if cfg.mode == "train":
        print("################## training #####################")
        train(cfg)
        # print("################## evaluating #####################")
        # evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train'], but got '{cfg.mode}'")


"""
python train.py -cn train.yaml \
    pre_output_path=/home/chenkai26/Paddle-AeroSimOpt/pre_output/dataset1 \
    train_ratio=0.5 \
    test_ratio=0.5 \
    task_id=1
"""
if __name__ == "__main__":
    main()
