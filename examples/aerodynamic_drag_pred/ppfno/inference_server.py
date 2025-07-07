#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved


from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from contextlib import asynccontextmanager
from timeit import default_timer
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import anyio
import hydra
import meshio
import numpy as np
import paddle
import pyvista as pv
import vtk
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from omegaconf import DictConfig
from paddle.distributed import ParallelEnv
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from pydantic import BaseModel

from ppcfd.models.ppfno.data import instantiate_inferencedatamodule
from ppcfd.models.ppfno.losses import LpLoss
from ppcfd.models.ppfno.networks import instantiate_network
from ppcfd.models.ppfno.optim.schedulers import instantiate_scheduler
from ppcfd.models.ppfno.utils.average_meter import AverageMeter
from ppcfd.models.ppfno.utils.average_meter import AverageMeterDict
from ppcfd.models.ppfno.utils.dot_dict import DotDict
from ppcfd.models.ppfno.utils.dot_dict import flatten_dict

# strategy = fleet.DistributedStrategy()
# strategy.find_unused_parameters = True
# fleet.init(is_collective=True, strategy=strategy)


# 设置你要的配置打印日志

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class InputData(BaseModel):
    # data_path: str  # /home/chenkai26/Paddle-AeroSimOpt/refine_data
    pre_output_path: str  # /aidsw01/paddlefile/hstasim/pre_output/{数据集id}
    reason_input_path: str  # /aidsw01/paddlefile/hstasim/pre_process/{case_id}
    reason_output_path: str  # /aidsw01/paddlefile/hstasim/reason_output/reason_{taskId}
    # state: str # /home/chenkai26/Paddle-AeroSimOpt/trainedModel/models/GNOFNOGNO_all_849.pdparams


class OutputData(BaseModel):
    error_code: int
    error_message: str
    cost_all: float
    cost_forward: float
    Cd_pred_modify: float
    pred_pressure_csv_path: str
    pred_pressure_vtp_path: str
    pred_wallshearstress_csv_path: str
    pred_wallshearstress_vtp_path: str


# 模型
MODEL: paddle.nn.Layer = None
CFG = None


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
            f"coord length({len(coord)}) should be an integer multiple of "
            f"num_timestamps({num_timestamps})"
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
                raise ValueError(
                    f"coord length({len(coord_)}) should be equal to value length({len(value_)})"
                )
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


# 模型加载函数
def load_model():
    global MODEL
    try:
        # TODO() 这里替换为实际的模型加载代码
        MODEL = instantiate_network(CFG)
        # loss_fn = LpLoss(size_average=True)
        if isinstance(MODEL, paddle.DataParallel):
            MODEL = MODEL._layers
        MODEL.eval()

        assert CFG.pd_path is not None, "checkpoint must be given."

        state = paddle.load(path=str(CFG.pd_path))
        MODEL.set_state_dict(state_dict=state["model"])
        device = ParallelEnv().device_id
        memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (
            1024 * 1024 * 1024
        )
        logging.info(f"Memory usage with model loading: {memory_allocated:.2f} GB")

        logging.info(f"Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


semaphore = None


# 应用生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    :param app:
    :return:
    """
    global semaphore
    # TODO() 设置并发数，例如2
    semaphore = asyncio.Semaphore(5)  # 在FastAPI启动时初始化
    # TODO() 启动时加载模型
    load_model()
    yield
    # 关闭时清理资源
    if MODEL is not None:
        # MODEL = None  # 或实际模型的清理代码
        pass


app = FastAPI(lifespan=lifespan)


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}


async def async_save_eval_results(
    cfg, pred, indices, caseid, decode_fn, output: OutputData
):
    try:
        (
            pred_pressure_csv_path,
            pred_pressure_vtp_path,
            pred_wallshearstress_csv_path,
            pred_wallshearstress_vtp_path,
        ) = save_eval_results(
            cfg,
            pred,
            indices[0],
            caseid,
            decode_fn=decode_fn,
        )
        
        # 更新输出对象中的文件路径
        output.pred_pressure_csv_path = pred_pressure_csv_path
        output.pred_pressure_vtp_path = pred_pressure_vtp_path
        output.pred_wallshearstress_csv_path = pred_wallshearstress_csv_path
        output.pred_wallshearstress_vtp_path = pred_wallshearstress_vtp_path
        
    except Exception as e:
        logging.error(f"Error in async file saving: {str(e)}")


# @app.post("/api/v1/inference", response_model=OutputData)
async def infer_model_task(input_data: InputData) -> OutputData:
    print(f"got input: {input_data}")
    global CFG
    CFG.pre_output_path = input_data.pre_output_path
    CFG.reason_input_path = input_data.reason_input_path
    CFG.reason_output_path = input_data.reason_output_path
    os.makedirs(os.path.join(CFG.reason_output_path, "log"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(CFG.reason_output_path, "log", "reason.txt"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
        force=True,
    )
    logging.info(f"开始处理请求(线程ID: {id(asyncio.get_running_loop())})")
    # await asyncio.sleep(5)
    with paddle.no_grad():
        global MODEL
        if MODEL is None:
            logging.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        try:
            datamodule = instantiate_inferencedatamodule(
                CFG, CFG.reason_input_path, CFG.pre_output_path, CFG.n_inference_num
            )
            inference_dataloader = datamodule.inference_dataloader(
                enable_ddp=CFG.enable_ddp, batch_size=CFG.batch_size
            )
            # all_files = os.listdir(os.path.join(CFG.reason_input_path, CFG.mode))
            all_files = os.listdir(CFG.reason_input_path)
            prefix = "area"
            indices = [item[5:9] for item in all_files if item.startswith(prefix)]

            def extract_number(s):
                return int(s)

            # create output json
            def create_json(json_file_path):
                if not os.path.exists(os.path.dirname(json_file_path)):
                    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

                if os.path.isfile(json_file_path):
                    os.remove(json_file_path)

                with open(json_file_path, "w") as file:
                    json.dump([], file)

            inference_json_file_path = os.path.join(
                CFG.reason_output_path,
                "json",
                "reason.json",
            )
            create_json(inference_json_file_path)

            def append_dict_to_json_list(file_path, dict_element):
                with open(file_path, "r") as file:
                    data = json.load(file)

                if isinstance(data, list):
                    data.append(dict_element)
                else:
                    logging.info("Error: The root of the JSON file is not a list.")
                    return
                with open(file_path, "w") as file:
                    json.dump(data, file, indent=4)

            indices.sort(key=extract_number)
            logging.info(f"Start evaluting {CFG.model} ...")
            eval_meter = AverageMeterDict()
            visualize_data_dicts = []
            loss_fn = LpLoss(size_average=True)

            def cal_mre(pred, label):
                return paddle.abs(x=pred - label) / paddle.abs(x=label)

            # for i, data_dict in enumerate(inference_dataloader):
            data_dict = next(iter(inference_dataloader))
            msg = ""
            inference_json_dict = {}
            device = ParallelEnv().device_id
            device = paddle.CUDAPlace(device)
            try:
                t1 = default_timer()
                out_dict, pred, cd_dict = MODEL.inference_dict(
                    device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
                )
                t2 = default_timer()
                paddle.device.cuda.empty_cache()
                msg += f"Inference (pure) took {t2 - t1:.2f} seconds."
                # logging.info('cd_dict:', cd_dict)
                (
                    pred_pressure_csv_path,
                    pred_pressure_vtp_path,
                    pred_wallshearstress_csv_path,
                    pred_wallshearstress_vtp_path,
                ) = get_pathes(
                    CFG,
                    datamodule.inference_full_caseids[0],
                )

                output = OutputData(
                    error_code=0,
                    error_message="",
                    cost_forward=t2 - t1,
                    cost_all=0.0,
                    Cd_pred_modify=cd_dict["Cd_pred_modify"],
                    pred_pressure_csv_path=pred_pressure_csv_path,
                    pred_pressure_vtp_path=pred_pressure_vtp_path,
                    pred_wallshearstress_csv_path=pred_wallshearstress_csv_path,
                    pred_wallshearstress_vtp_path=pred_wallshearstress_vtp_path,
                )
                if CFG.save_eval_results:

                    # (
                    #     pred_pressure_csv_path,
                    #     pred_pressure_vtp_path,
                    #     pred_wallshearstress_csv_path,
                    #     pred_wallshearstress_vtp_path,
                    # ) = save_eval_results(
                    #     CFG,
                    #     pred,
                    #     indices[0],
                    #     datamodule.inference_full_caseids[0],
                    #     decode_fn=datamodule.decode,
                    # )

                    asyncio.create_task(
                        async_save_eval_results(
                            CFG,
                            pred,
                            indices[0],
                            datamodule.inference_full_caseids[0],
                            datamodule.decode,
                            output
                        )
                    )

            except MemoryError as e:
                logging.info(e)
                if "Out of memory" in str(e):
                    logging.info(f"WARNING: OOM on sample {0}, skipping this sample.")
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    # continue
                else:
                    raise

            msg += f"Eval sample {0}... L2_Error: "
            for k, v in out_dict.items():
                if k.split("_")[0] == "L2":
                    msg += f"{k}: {v.item():.4f}, "
                    eval_meter.update({k: v})
            msg += f"|| MRE and Value: "
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
            # Cd_pred = out_dict["Cd_pred"].item()
            # Cd_mre_modify = paddle.abs(x=Cd_pred_modify - Cd_truth) / paddle.abs(x=Cd_truth)
            # eval_meter.update({"Cd_mre_modify": Cd_mre_modify})
            eval_meter.update({"Cd_pred_modify": Cd_pred_modify})

            # msg += f"MRE_Cd_modify: {Cd_mre_modify.item():.4f}, "
            msg += f"Cd_pred_modify: {Cd_pred_modify.item():.4f}, "
            # msg += f"Cd_truth: {Cd_truth.item():.4f}], "

            inference_json_dict["parts"] = os.path.basename(CFG.reason_input_path)
            inference_json_dict["drag_coefficient"] = Cd_pred_modify.item()
            inference_json_dict["pressure_drag_coefficient"] = (
                cd_dict["Cd_pressure_pred"]
                + cd_dict["Cd_pred_modify"].item()
                - out_dict["Cd_pred"].item()
            )

            inference_json_dict["friction_resistance_coefficient"] = cd_dict[
                "Cd_wallshearstress_pred"
            ]
            inference_json_dict["total_drag"] = cd_dict["total_drag_pred"]
            inference_json_dict["pressure_drag"] = cd_dict["pressure_drag_pred"]
            inference_json_dict["friction_resistance"] = cd_dict[
                "wallshearstress_drag_pred"
            ]
            t3 = default_timer()

            inference_json_dict["cost_all"] = t3 - t1
            inference_json_dict["cost_forward"] = t2 - t1
            append_dict_to_json_list(inference_json_file_path, inference_json_dict)

            logging.info(msg)

            msg = f"Inference took {t3 - t1:.2f} seconds. Everage eval values: "
            eval_dict = eval_meter.avg
            for k, v in eval_dict.items():
                msg += f"{v.item():.4f}({k}), "
            logging.info(msg)
            max_memory_allocated = paddle.device.cuda.max_memory_allocated(
                device=device
            ) / (1024 * 1024 * 1024)
            logging.info(f"Memory Usage: {max_memory_allocated:.2f} GB (MAX).")

            # output = OutputData(
            #     error_code=0,
            #     error_message="",
            #     cost_forward=t2 - t1,
            #     cost_all=t3 - t1,
            #     Cd_pred_modify=Cd_pred_modify,
            #     pred_pressure_csv_path=pred_pressure_csv_path,
            #     pred_pressure_vtp_path=pred_pressure_vtp_path,
            #     pred_wallshearstress_csv_path=pred_wallshearstress_csv_path,
            #     pred_wallshearstress_vtp_path=pred_wallshearstress_vtp_path,
            # )
            logging.info(f"请求处理完成(线程ID: {id(asyncio.get_running_loop())})")
            return output
        except Exception as e:
            logging.error(f"请求处理出现错误(线程ID: {id(asyncio.get_running_loop())})")
            logging.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )


@app.post("/api/v1/inference", response_model=OutputData)
async def infer_model(input_data: InputData) -> OutputData:
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    # logging.info("got input:{}".format(input_data.input_file))
    try:
        async with semaphore:
            # TODO() 这里设置单个请求的超时时间，业务层超时
            result = await asyncio.wait_for(infer_model_task(input_data), timeout=60)
            return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="request timeout > 60s",
        )
    except Exception as e:
        # logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


def save_eval_results(
    cfg: DictConfig, pred, centroid_idx, caseid, decode_fn=None
) -> Tuple[str, str, str]:
    pred_pressure = decode_fn(pred[0:1, :], 0).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1).cpu().detach().numpy()
    evals_results = {
        "pred_pressure": pred_pressure,
        "pred_wallshearstress": pred_wallshearstress,
    }
    centroid = np.load(f"{cfg.reason_input_path}/centroid_{centroid_idx}.npy")
    cells = [("vertex", np.arange(tuple(centroid.shape)[0]).reshape(-1, 1))]

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


def get_pathes(
    cfg: DictConfig, caseid
) -> Tuple[str, str, str, str]:
    # pred_pressure = decode_fn(pred[0:1, :], 0).cpu().detach().numpy()
    # pred_wallshearstress = decode_fn(pred[1:4, :], 1).cpu().detach().numpy()
    evals_results = {
        "pred_pressure": None,
        "pred_wallshearstress": None,
    }

    pred_pressure_csv_path = None
    pred_pressure_vtp_path = None
    pred_wallshearstress_csv_path = None
    pred_wallshearstress_vtp_path = None

    for k, v in evals_results.items():
        csv_filename = os.path.join(
            cfg.reason_output_path,
            "vtp_csv",
            f"{caseid}_{k}.csv",
        )
        vtp_filename = os.path.join(
            cfg.reason_output_path,
            "vtp_csv",
            f"{caseid}_{k}.vtp",
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
    global CFG
    CFG = cfg
    import uvicorn

    port = os.getenv("main", "8087")
    uvicorn.run(app, host="0.0.0.0", workers=1, port=int(port))

    """
    CUDA_VISIBLE_DEVICES=1 python inference_server.py \
        -cn inference.yaml pd_path=/home/chenkai26/Paddle-AeroSim-DataModel/checkpoints/GNOFNOGNO_all.pdparams
    """

    """
    curl -X POST "http://0.0.0.0:8087/api/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"reason_output_path":"/home/chenkai26/Paddle-AeroSim-DataModel/inference_output/dataset1/case016", "reason_input_path":"/home/chenkai26/Paddle-AeroSim-DataModel/pre_output/inference_dataset1/case016", "pre_output_path":"/home/chenkai26/Paddle-AeroSim-DataModel/pre_output/train_dataset1"}'
    """

    """
    task_id: 123
    pd_path: /aidsw01/paddlefile/hstasim/train_output/train_{taskId}/pd/GNOFNOGNO_all_849.pdparams，
    pre_output_path:/aidsw01/paddlefile/hstasim/pre_output/{数据集id}  # 拿txt文件
    reason_input_path:/aidsw01/paddlefile/hstasim/pre_process/{case_id},
    reason_output_path:/aidsw01/paddlefile/hstasim/reason_output/reason_{taskId},
    gpu: 1,2
    """


if __name__ == "__main__":
    main()
