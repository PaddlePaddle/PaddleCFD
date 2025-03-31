import os

import paddle
from datamodules import PhysicalDataLoader, PhysicalDatastet
from omegaconf import DictConfig, OmegaConf


def get_data_dir(
    root_data_dir: str,
    physical_system: str = "navier-stokes",
    num_test_obstacles: int = 1,
    test_out_of_distribution: bool = False,
    **kwargs,
):
    ood_infix = "outdist-" if test_out_of_distribution else ""
    if physical_system == "navier-stokes":
        _first_subdir = "navier-stokes-multi"
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

    # self.root_data_dir = os.path.join(self.root_data_dir, _first_subdir)
    _first_subdir = os.path.join(_first_subdir, "run", "data_gen")
    data_dir_train = os.path.join(root_data_dir, _first_subdir, subdirs["train"])
    data_dir_val = os.path.join(root_data_dir, _first_subdir, subdirs["val"])
    data_dir_test = os.path.join(root_data_dir, _first_subdir, subdirs["test"])
    return {"train": data_dir_train, "val": data_dir_val, "test": data_dir_test}


def get_dataloader(cfg_data: DictConfig, modes=["train"]):
    # get datadirs
    data_dir_dict = get_data_dir(root_data_dir=cfg_data.root_data_dir, **cfg_data.dataset)
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

    if opt_path and os.path.exists(f"{opt_path}.pdopt"):
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
