import logging
import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import paddle
from datamodules import PhysicalDataLoader
from datamodules import PhysicalDatastet
from matplotlib.animation import FuncAnimation
from omegaconf import DictConfig
from omegaconf import OmegaConf

from ppcfd.models import ppdiffusion as models


# Random seed (if not using pypaddle-lightning)
def set_seed(seed, device="gpu"):
    """
    Sets the random seed for the given device.
    """
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    if device != "cpu" and paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")


def get_data_path(
    root_data_dir: str,
    physical_system: str = "navier-stokes",
    num_test_obstacles: int = 1,
    test_out_of_distribution: bool = False,
    **kwargs,
):
    ood_infix = "outdist-" if test_out_of_distribution else ""
    if physical_system == "navier-stokes":
        # _first_subdir = "navier-stokes-multi"
        assert num_test_obstacles in [
            1,
            4,
        ], f"Invalid number of test obstacles {num_test_obstacles}"
        test_t = {(1): 65, (4): 16, (16): 4}[num_test_obstacles]
        test_set_name = f"ns-runs_eval-{ood_infix}cors{num_test_obstacles}-navier-stokes-n5-t{test_t}-n0_tagcors{num_test_obstacles}_00001"
        subdirs = {
            "train": "ns-runs_train-navier-stokes-n100-t65-n0_00001",
            "val": "ns-runs_val-navier-stokes-n2-t65-n0_00001",
            "test": test_set_name,
        }
        subdirs["predict"] = subdirs["val"]
    else:
        raise NotImplementedError(f"Physical system {physical_system} is not implemented yet.")

    data_dir_train = os.path.join(root_data_dir, subdirs["train"])
    data_dir_val = os.path.join(root_data_dir, subdirs["val"])
    data_dir_test = os.path.join(root_data_dir, subdirs["test"])
    return {"train": data_dir_train, "val": data_dir_val, "test": data_dir_test}


def get_dataloader(cfg_data: DictConfig, modes=["train", "val", "test"]):
    # get datadirs
    data_dir_dict = get_data_path(root_data_dir=cfg_data.root_data_dir, **cfg_data.dataset)
    # get batch_size
    cfg_dataloader = OmegaConf.to_container(cfg_data.dataloader, resolve=True)
    batch_size = cfg_dataloader.pop("batch_size", None)

    # get dataloaders
    dataloaders = {}
    for mode in modes:
        shuffle = True if mode == "train" else False
        dataset = PhysicalDatastet(data_dir=data_dir_dict[mode], **cfg_data.dataset)
        dataloaders[mode] = PhysicalDataLoader(dataset).dataloader(
            **cfg_dataloader, batch_size=batch_size[mode], shuffle=shuffle
        )

    return dataloaders


def get_optimizer(cfg_opt: DictConfig, parameters, opt_path=None):
    cfg_opt = OmegaConf.to_container(cfg_opt, resolve=True)
    opt_name = cfg_opt.pop("name", None)
    if opt_name.lower() == "adamw":
        optim_class = paddle.optimizer.AdamW
    elif opt_name.lower() == "adam":
        optim_class = paddle.optimizer.Adam
    else:
        raise ValueError(f"Optimizer {opt_name} not supported now.")

    # set grad_clip
    grad_clip = cfg_opt.pop("grad_clip", None)
    if grad_clip:
        clip_al, clip_val = grad_clip
        if clip_al == "global_norm":
            grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_val)
        elif clip_al == "norm":
            grad_clip = paddle.nn.ClipGradByNorm(clip_val)
        elif clip_al == "value":
            if isinstance(clip_val, float):
                clip_val = [-clip_val, clip_val]
            grad_clip = paddle.nn.ClipGradByValue(max=clip_val[1], min=clip_val[0])
        else:
            raise ValueError(f"Gradient clipping algorithm {clip_al} not supported now.")
    cfg_opt["grad_clip"] = grad_clip

    optimizer = optim_class(parameters=parameters, **cfg_opt)

    if opt_path:
        if opt_path.endswith(".pdparams"):
            opt_path = opt_path.replace(".pdparams", ".pdopt")
        elif not opt_path.endswith(".pdopt"):
            opt_path = f"{opt_path}.pdopt"
        if os.path.exists(f"{opt_path}.pdopt"):
            optim_dict = paddle.load(f"{opt_path}.pdopt")
            optimizer.set_state_dict(optim_dict)
    return optimizer


def get_scheduler(cfg_sched: DictConfig, optimizer):
    cfg_sched = OmegaConf.to_container(cfg_sched, resolve=True)
    sched_name = cfg_sched.pop("name", None)
    if sched_name.lower() == "cosine":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay
    elif sched_name.lower() == "linear":
        scheduler = paddle.optimizer.lr.LinearWarmupLR
    else:
        raise ValueError(f"Scheduler {sched_name} not supported now.")
    return scheduler(**cfg_sched)


def dynamic_import(model_name: str):
    try:
        ModelClass = getattr(models, model_name)
    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models")
    return ModelClass


def initialize_models(cfg, models_lst=["interp"], interp_obj=None, forecast_obj=None):
    init_models = []

    if "interp" in models_lst:
        assert interp_obj is not None, "interp_obj should not be None if 'interp' in models_lst"
        # load interpolation model
        interp_model = dynamic_import(cfg.INTERPOLATION.MODEL.model_name)
        model_interp_cfg = interp_obj.model_cfg_transform(cfg.INTERPOLATION.MODEL)
        model_interp = interp_model(**model_interp_cfg)
        # load ckpt
        ckpt = getattr(cfg.INTERPOLATION, "checkpoint", None)
        if ckpt:
            ckpt_path = ckpt if ckpt.endswith(".pdparams") else f"{ckpt}.pdparams"
            state_dict = paddle.load(ckpt_path)
            model_interp.set_state_dict(state_dict)
            logging.info(f"Finish loading checkpoint {ckpt_path}")
        init_models.append(model_interp)

    if "forecast" in models_lst:
        assert forecast_obj is not None, "forecast_obj should not be None if 'forecast' in models_lst"
        # init forecasting model
        forecast_model = dynamic_import(cfg.FORECASTING.MODEL.model_name)
        model_forecast_cfg = forecast_obj.model_cfg_transform(cfg.FORECASTING.MODEL)
        model_forecast = forecast_model(**model_forecast_cfg)
        # load ckpt
        ckpt = getattr(cfg.FORECASTING, "checkpoint", None)
        if ckpt:
            ckpt_path = ckpt if ckpt.endswith(".pdparams") else f"{ckpt}.pdparams"
            state_dict = paddle.load(ckpt_path)
            model_forecast.set_state_dict(state_dict)
            logging.info(f"Finish loading checkpoint {ckpt_path}")
        init_models.append(model_forecast)

    return init_models


def save_arrays_as_line_plot(
    x_array: np.ndarray,
    key_to_array: Dict[str, np.ndarray],
    save_dir: str = "./plots",
    x_label: str = "x",
    y_label: str = "y",
    figsize: tuple = (10, 6),
    dpi: int = 300,
    format: str = "png",
    show_grid: bool = True,
    line_style: str = "-",
    extra_info: str = "",
):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{x_label}_vs_{y_label}{extra_info}.{format}"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=figsize, dpi=dpi)

    for label, y_array in key_to_array.items():
        if isinstance(y_array, paddle.Tensor):
            y_array = y_array.numpy()
        plt.plot(x_array, y_array, line_style, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    if show_grid:
        plt.grid(True)

    plt.savefig(save_path, bbox_inches="tight", format=format)
    plt.close()
    logging.info(f"Plot saved to: {save_path}")


def save_arrays_as_gif(
    preds,
    targets,
    save_dir: str = "./plots",
    titles=["u", "v", "p"],
    extra_info: str = "",
):
    logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)
    os.makedirs(save_dir, exist_ok=True)

    T, B, C, _, _ = targets.shape
    assert C == len(titles), f"Error: targets' features is {C} != titles' num {len(titles)}."

    preds = preds.transpose(0, 1, 2, 4, 3)
    targets = targets.transpose(0, 1, 2, 4, 3)
    diffs = np.abs(preds - targets)

    for f in range(C):
        fig, axs = plt.subplots(3, 1, figsize=(9, 6), constrained_layout=True)
        fig.suptitle(f"Frame: 0/{T} ({titles[f]}) of Pred/Target/Diff", fontsize=14)

        gif_data = [preds[:, 0, f], targets[:, 0, f], diffs[:, 0, f]]
        combined_data = np.concatenate([arr.flatten() for arr in gif_data])
        data_min, data_max = np.percentile(combined_data, [1, 99])
        norm = plt.Normalize(vmin=data_min, vmax=data_max)

        imgs = []
        for row in range(3):
            ax = axs[row]
            img = ax.imshow(gif_data[row][0], norm=norm, cmap="coolwarm", aspect="auto", interpolation="bilinear")
            imgs.append(img)

        cbar = fig.colorbar(imgs[0], ax=axs, orientation="vertical", pad=0.02)
        cbar.set_label("Normalized Value")

        def update(t):
            for i, img in enumerate(imgs):
                img.set_data(gif_data[i][t])
                img.set_norm(norm)
            fig.suptitle(f"Frame: {t+1}/{T} ({titles[f]}) of Pred/Target/Diff", fontsize=14)
            return imgs

        def init():
            for i, img in enumerate(imgs):
                img.set_data(gif_data[i][0])
            return imgs

        ani = FuncAnimation(
            fig,
            update,
            frames=T,
            interval=500,
            blit=True,
            repeat_delay=1000,
            init_func=init,
        )

        filename = f"{titles[f]}{extra_info}.gif"
        save_path = os.path.join(save_dir, filename)
        ani.save(save_path, writer="pillow", fps=2, dpi=100, savefig_kwargs={"facecolor": "white"})
        plt.close(fig)
        logging.info(f"GIF saved to: {save_path}")
