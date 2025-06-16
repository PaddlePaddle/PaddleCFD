import json
import logging
import os
import sys

sys.path.append("./src")
sys.path.append("./src/networks")
from timeit import default_timer
from typing import Dict, Tuple

import hydra
import numpy as np
import paddle
import pyvista as pv
from omegaconf import DictConfig
from paddle import distributed as dist
from paddle.distributed import ParallelEnv, fleet
from src.data import instantiate_inferencedatamodule
from src.losses import LpLoss
from src.networks import instantiate_network
from src.utils.average_meter import AverageMeterDict

os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
# os.environ["HYDRA_FULL_ERROR"] = "0"


# strategy = fleet.DistributedStrategy()
# strategy.find_unused_parameters = True
# fleet.init(is_collective=True, strategy=strategy)


def set_seed(seed: int = 0):
    paddle.seed(seed=seed)
    np.random.seed(seed)


world_size = dist.get_world_size()
if world_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.find_unused_parameters = True
    fleet.init(is_collective=True, strategy=strategy)


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


@paddle.no_grad()
def inference(cfg: DictConfig):
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # init logger
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"/{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
    )

    # inference_json_file_path = os.path.join('/home/chenkai26/Paddle-AeroSimOpt/output/dataset1/inference/case1', 'inference.json')
    # "/home/chenkai26/Paddle-AeroSimOpt/refine_data/dataset1/inference/2014-f-6103"
    inference_json_file_path = os.path.join(
        cfg.reason_output_path,
        "json",
        "reason.json",
    )

    def create_json(json_file_path):
        if not os.path.exists(os.path.dirname(json_file_path)):
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

        if os.path.isfile(json_file_path):
            os.remove(json_file_path)

        with open(json_file_path, "w") as file:
            json.dump([], file)

    create_json(inference_json_file_path)

    def append_dict_to_json_list(file_path, dict_element):
        with open(file_path, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            data.append(dict_element)
        else:
            print("Error: The root of the JSON file is not a list.")
            return
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    # init model
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

    # run prediction
    datamodule = instantiate_inferencedatamodule(cfg, cfg.reason_input_path, cfg.pre_output_path, cfg.n_inference_num)
    inference_dataloader = datamodule.inference_dataloader(enable_ddp=cfg.enable_ddp, batch_size=cfg.batch_size)
    # all_files = os.listdir(os.path.join(cfg.data_path, cfg.mode))
    # all_files = os.listdir(cfg.reason_input_path)
    # prefix = "area"
    # indices = [item[5:9] for item in all_files if item.startswith(prefix)]

    # def extract_number(s):
    #     return int(s)

    # indices.sort(key=extract_number)
    logging.info(f"Start evaluting {cfg.model} ...")

    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.eval()
    eval_meter = AverageMeterDict()

    def cal_mre(pred, label):
        return paddle.abs(x=pred - label) / paddle.abs(x=label)

    for i, data_dict in enumerate(inference_dataloader):
        inference_json_dict = {}
        device = ParallelEnv().device_id
        device = paddle.CUDAPlace(device)
        try:
            t1 = default_timer()
            out_dict, pred, cd_dict = model.inference_dict(
                device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
            )
            t2 = default_timer()
            print(f"Inference {i} costs: {t2 - t1:.2f} seconds.")
            # print('cd_dict:', cd_dict)
            if cfg.save_eval_results:
                save_eval_results(
                    cfg,
                    pred,
                    datamodule.inference_indices[0],
                    datamodule.inference_full_caseids[0],
                    decode_fn=datamodule.decode,
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
        """
        for k, v in out_dict.items():
            if "Cd" and "pred" in k.split("_"):
                k_truth = f"{k[:k.rfind('_')]}_truth"
                mre = cal_mre(v, out_dict[k_truth])
                eval_meter.update({f"MRE_{k[:k.rfind('_')]}": mre})
                msg += f"MRE_{k[:k.rfind('_')]}: {mre.item():.4f}, "
                msg += f"[{k}: {v:.4f}, {k_truth}: {out_dict[k_truth]:.4f}], "
        """
        Cd_pred_modify = cd_dict["Cd_pred_modify"]
        # Cd_truth = out_dict["Cd_truth"]
        out_dict["Cd_pred"]
        # Cd_mre_modify = paddle.abs(x=Cd_pred_modify - Cd_truth) / paddle.abs(x=Cd_truth)
        # eval_meter.update({"Cd_mre_modify": Cd_mre_modify})
        eval_meter.update({"Cd_pred_modify": Cd_pred_modify})

        # msg += f"MRE_Cd_modify: {Cd_mre_modify.item():.4f}, "
        msg += f"[Cd_pred_modify: {Cd_pred_modify.item():.4f}, "
        # msg += f"Cd_truth: {Cd_truth.item():.4f}], "

        inference_json_dict["Cd_pred"] = Cd_pred_modify.item()
        inference_json_dict["Cd_pressure_pred"] = cd_dict["Cd_pressure_pred"].item()
        inference_json_dict["Cd_wallshearstress_pred"] = cd_dict["Cd_wallshearstress_pred"].item()
        inference_json_dict["total_drag_pred"] = cd_dict["total_drag_pred"].item()
        inference_json_dict["pressure_drag_pred"] = cd_dict["pressure_drag_pred"].item()
        inference_json_dict["wallshearstress_drag_pred"] = cd_dict["wallshearstress_drag_pred"].item()
        append_dict_to_json_list(inference_json_file_path, inference_json_dict)

        logging.info(msg)

    t3 = default_timer()
    msg = f"Inference + vtp file saving took {t3 - t1:.2f} seconds. Everage eval values: "
    eval_dict = eval_meter.avg
    for k, v in eval_dict.items():
        msg += f"{v.item():.4f}({k}), "
    logging.info(msg)
    max_memory_allocated = paddle.device.cuda.max_memory_allocated(device=device) / (1024 * 1024 * 1024)
    print(f"Memory Usage: {max_memory_allocated:.2f} GB (MAX).")


def save_eval_results(cfg: DictConfig, pred, centroid_idx, caseid, decode_fn=None) -> Tuple[str, str, str]:
    pred_pressure = decode_fn(pred[0:1, :], 0).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1).cpu().detach().numpy()
    evals_results = {
        "pred_pressure": pred_pressure,
        "pred_wallshearstress": pred_wallshearstress,
    }
    centroid = np.load(f"{cfg.reason_input_path}/centroid_{centroid_idx}.npy")

    centroid = centroid[:: cfg.subsample_eval, ...]

    [("vertex", np.arange(tuple(centroid.shape)[0]).reshape(-1, 1))]

    os.makedirs(os.path.join(cfg.reason_output_path, "vtp_csv"), exist_ok=True)

    pred_pressure_csv_path = None
    pred_pressure_vtp_path = None
    pred_wallshearstress_csv_path = None
    pred_wallshearstress_vtp_path = None

    logging.info(evals_results.keys())
    for k, v in evals_results.items():
        # print(k, v.shape)
        # np.save(
        # os.path.join(f"{cfg.reason_input_path}/evals_results", f"{k}_{centroid_idx}.npy"),
        # v.T,
        # )
        # save 6 csv output files
        array_hstack = np.hstack((centroid, v.T))
        csv_filename = os.path.join(
            # "/home/chenkai26/Paddle-AeroSimOpt/output/dataset1/inference/case1",
            cfg.reason_output_path,
            "vtp_csv",
            f"{caseid}_{k}.csv",
        )
        np.savetxt(csv_filename, array_hstack, delimiter=",", fmt="%f")

        # save 6 vtp output files
        # mesh = meshio.Mesh(points=centroid, cells=cells)
        # mesh.point_data.update({f"{k}": v.T})
        vtp_filename = os.path.join(
            # "/home/chenkai26/Paddle-AeroSimOpt/output/dataset1/inference/case1",
            # "/home/chenkai26/Paddle-AeroSimOpt/output/dataset1/inference/case1",
            cfg.reason_output_path,
            "vtp_csv",
            f"{caseid}_{k}.vtp",
        )
        # mesh.write(vtp_filename, file_format="vtk", binary=False)
        # legacy_to_xml(vtp_filename)
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

        if k == "pred_pressure":
            pred_pressure_csv_path = csv_filename
            pred_pressure_vtp_path = vtp_filename
        elif k == "pred_wallshearstress":
            pred_wallshearstress_csv_path = csv_filename
            pred_wallshearstress_vtp_path = vtp_filename

    return (
        pred_pressure_csv_path,
        pred_pressure_vtp_path,
        pred_wallshearstress_csv_path,
        pred_wallshearstress_vtp_path,
    )


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(cfg: DictConfig):
    inference(cfg)


if __name__ == "__main__":
    main()
