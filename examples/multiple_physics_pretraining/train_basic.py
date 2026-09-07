import argparse
import gc
import os
import pickle as pkl
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext

# # 打开组合算子
# export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true

# # 打开 CINN 编译器
# export FLAGS_use_cinn=true

# # 是否打印 Program IR 信息 (用于调试)
# export FLAGS_print_ir=false

# 单一 CINN 开关：MPP_USE_CINN（默认 "0"，不启用）。
#   开 → 显式动转静 + CINN flags
#   关 → 原版纯动态
_MPP_USE_CINN = os.environ.get("MPP_USE_CINN", "0") == "1"
if _MPP_USE_CINN:
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"
os.environ.setdefault("FLAGS_print_ir", "false")

import einops
import numpy as np
import paddle
import wandb


# from adan_pytorch import Adan
# from dadaptation import DAdaptAdam, DAdaptAdan
from ppcfd.models.multiple_physics_pretraining.paddle_utils import *
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from ppcfd.models.multiple_physics_pretraining.utils.adan_paddle import Adan
from ppcfd.models.multiple_physics_pretraining.utils.dadapt_adam_paddle import (
    DAdaptAdam,
)
from ppcfd.models.multiple_physics_pretraining.utils.dadapt_adan_paddle import (
    DAdaptAdan,
)


# from torchinfo import summary

from ppcfd.models.multiple_physics_pretraining.data_utils.datasets import (
    DSET_NAME_TO_OBJECT,
    get_data_loader,
)
from ppcfd.models.multiple_physics_pretraining.avit import build_avit
from ppcfd.models.multiple_physics_pretraining.utils import logging_utils
from ppcfd.models.multiple_physics_pretraining.utils.schedulers import (
    SimpleSequentialScheduler,
    dadapt_cosine_scheduler,
)
from ppcfd.models.multiple_physics_pretraining.utils.YParams import YParams


def add_weight_decay(model, weight_decay=1e-05, inner_lr=0.001, skip_list=()):
    """From Ross Wightman at:
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

    Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale)
    then don't apply weight decay.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
        if len(param.squeeze().shape) <= 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class Trainer:
    def __init__(self, params, global_rank, local_rank, device, sweep_id=None):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(paddle.distributed.get_world_size())
        self.sweep_id = sweep_id
        self.log_to_screen = params.log_to_screen
        self.train_loss = paddle.nn.MSELoss()
        self.startEpoch = 0
        self.epoch = 0
        has_cuda_device = (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
        self.amp_enabled = has_cuda_device and params.enable_amp
        self.mp_type = (
            "bfloat16"
            if has_cuda_device and paddle.amp.is_bfloat16_supported()
            else "float16"
        )
        self.iters = 0
        self.initialize_data(self.params)
        print(f"Initializing model on rank {self.global_rank}")
        self.initialize_model(self.params)
        self.initialize_optimizer(self.params)
        if params.resuming:
            print("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)
        if params.resuming == False and params.pretrained:
            print("Starting from pretrained model at %s" % params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.startEpoch = 0
        # 动转静
        if os.environ.get("MPP_USE_CINN", "0") == "1":
            full_graph = getattr(self.params, "to_static_full_graph", True)
            self.single_print(
                f"[to_static] wrapping AViT with paddle.jit.to_static, full_graph={full_graph}"
            )
            self.model = paddle.jit.to_static(self.model, full_graph=full_graph)
        self.initialize_scheduler(self.params)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(" ".join([str(t) for t in text]))

    def initialize_data(self, params):
        if params.tie_batches:
            in_rank = 0
        else:
            in_rank = self.global_rank
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")
        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            params,
            params.train_data_paths,
            paddle.distributed.is_initialized(),
            split="train",
            rank=in_rank,
            train_offset=self.params.embedding_offset,
        )
        self.valid_data_loader, self.valid_dataset, _ = get_data_loader(
            params,
            params.valid_data_paths,
            paddle.distributed.is_initialized(),
            split="val",
            rank=in_rank,
        )
        if paddle.distributed.is_initialized():
            self.train_sampler.set_epoch(0)

    def initialize_model(self, params):
        if self.params.model_type == "avit":
            self.model = build_avit(params).to(self.device)
        """
        # there's no match api for torch.compile in paddle.
        if self.params.compile:
            print(
                "WARNING: BFLOAT NOT SUPPORTED IN SOME COMPILE OPS SO SWITCHING TO FLOAT16"
            )
            self.mp_type = torch.half
            self.model = torch.compile(self.model)
        """
        if paddle.distributed.is_initialized():
            """paddle.DataParallel无device_ids和output_device"""
            self.model = paddle.DataParallel(
                layers=self.model, find_unused_parameters=True
            )
        self.single_print(
            f"Model parameter count: {sum([p.size for p in self.model.parameters()])}"
        )

    def initialize_optimizer(self, params):
        parameters = add_weight_decay(self.model, self.params.weight_decay)
        if params.optimizer == "adam":
            if self.params.learning_rate < 0:
                self.optimizer = DAdaptAdam(
                    parameters, lr=1.0, growth_rate=1.05, log_every=100, decouple=True
                )
            else:
                self.optimizer = paddle.optimizer.AdamW(
                    parameters=parameters,
                    learning_rate=params.learning_rate,
                    weight_decay=0.0,
                )
        elif params.optimizer == "adan":
            if self.params.learning_rate < 0:
                self.optimizer = DAdaptAdan(
                    parameters, lr=1.0, growth_rate=1.05, log_every=100
                )
            else:
                self.optimizer = Adan(parameters, lr=params.learning_rate)
        elif params.optimizer == "sgd":
            """there's no param "momentum" in paddle.optimizer.SGD
            and the param "lr" is named as "learning_rate", "params" is "parameters" in paddle.
            """
            self.optimizer = paddle.optimizer.SGD(
                parameters=self.model.parameters(),
                learning_rate=params.learning_rate,
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Optimizer {params.optimizer} not supported")
        self.gscaler = paddle.amp.GradScaler(
            enable=self.amp_enabled and self.mp_type == "float16",
            incr_every_n_steps=2000,
            init_loss_scaling=65536.0,
        )

    def initialize_scheduler(self, params):
        if params.scheduler_epochs > 0:
            sched_epochs = params.scheduler_epochs
        else:
            sched_epochs = params.max_epochs
        if params.scheduler == "cosine":
            if self.params.learning_rate < 0:
                tmp_lr = dadapt_cosine_scheduler(
                    self.optimizer,
                    sched_epochs * params.epoch_size,
                    last_epoch=self.startEpoch * params.epoch_size - 1,
                )
                self.optimizer.set_lr_scheduler(tmp_lr)
                self.scheduler = tmp_lr
            else:
                k = params.warmup_steps
                last_step = self.startEpoch * params.epoch_size - 1
                tmp_lr = paddle.optimizer.lr.LinearLR(
                    start_factor=0.01,
                    end_factor=1.0,
                    total_steps=k,
                    last_epoch=last_step,
                    learning_rate=self.optimizer.get_lr(),
                )
                warmup = tmp_lr
                tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
                    eta_min=params.learning_rate / 100,
                    T_max=sched_epochs * params.epoch_size - k,
                    last_epoch=last_step - k if last_step >= k else -1,
                    learning_rate=self.optimizer.get_lr(),
                )
                decay = tmp_lr
                self.scheduler = SimpleSequentialScheduler(
                    self.optimizer,
                    [warmup, decay],
                    [k],
                    last_epoch=params.epoch_size * self.startEpoch - 1,
                )
        else:
            self.scheduler = None

    def save_checkpoint(self, checkpoint_path, model=None):
        """Save model and optimizer to checkpoint"""
        if not model:
            model = self.model
        paddle.save(
            obj={
                "iters": self.epoch * self.params.epoch_size,
                "epoch": self.epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path=checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        """Load model/opt from path"""
        checkpoint = paddle.load(path=str(checkpoint_path))
        if "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            model_state = checkpoint
        try:
            self.model.load_state_dict(model_state)
        except:
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(model_state)
            else:
                new_state_dict = OrderedDict()
                for key, val in model_state.items():
                    name = key[7:]
                    new_state_dict[name] = val
                self.model.load_state_dict(new_state_dict)
        if self.params.resuming:
            self.iters = checkpoint["iters"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.startEpoch = checkpoint["epoch"]
            self.epoch = self.startEpoch
        else:
            self.iters = 0
        if self.params.pretrained:
            inner = getattr(self.model, "module", self.model)
            if getattr(self.params, "freeze_middle", False):
                inner.freeze_middle()
            elif getattr(self.params, "freeze_processor", False):
                inner.freeze_processor()
            else:
                inner.unfreeze()
            append = getattr(self.params, "append_datasets", None) or []
            if append:
                exp_proj = sum(
                    len(DSET_NAME_TO_OBJECT[add_on]._specifics()[2]) for add_on in append
                )
                inner.expand_projections(exp_proj)
        checkpoint = None
        self.model = self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        self.epoch += 1
        tr_time = 0
        data_time = 0
        data_start = time.time()
        self.model.train()
        logs = {
            "train_rmse": paddle.zeros(1).to(self.device),
            "train_nrmse": paddle.zeros(1).to(self.device),
            "train_l1": paddle.zeros(1).to(self.device),
        }
        steps = 0
        last_grads = [paddle.zeros_like(p) for p in self.model.parameters()]
        grad_logs = defaultdict(lambda: paddle.zeros(1).to(self.device))
        grad_counts = defaultdict(lambda: paddle.zeros(1).to(self.device))
        loss_logs = defaultdict(lambda: paddle.zeros(1).to(self.device))
        loss_counts = defaultdict(lambda: paddle.zeros(1).to(self.device))
        bench_fwd, bench_bwd, bench_opt, bench_step, bench_loss = [], [], [], [], []
        last_log_nrmse = float("nan")
        self.single_print(
            "train_loader_size", len(self.train_data_loader), len(self.train_dataset)
        )
        for batch_idx, data in enumerate(self.train_data_loader):
            steps += 1
            inp, file_index, field_labels, bcs, tar = map(
                lambda x: x.to(self.device), data
            )
            dset_type = self.train_dataset.sub_dsets[file_index[0]].type
            loss_counts[dset_type] += 1
            inp = einops.rearrange(inp, "b t c h w -> t b c h w")
            data_time += time.time() - data_start
            dtime = time.time() - data_start
            should_sync = (1 + batch_idx) % self.params.accum_grad == 0
            sync_context = nullcontext()
            if paddle.distributed.is_initialized() and not should_sync:
                sync_context = self.model.no_sync()
            with sync_context:
                with paddle.amp.auto_cast(enable=self.amp_enabled, dtype=self.mp_type):
                    model_start = time.time()
                    output = self.model(inp, field_labels, bcs)
                    spatial_dims = tuple(range(output.ndim))[2:]
                    residuals = output - tar
                    tar_norm = 1e-07 + tar.pow(2).mean(spatial_dims, keepdim=True)
                    raw_loss = residuals.pow(2).mean(
                        spatial_dims, keepdim=True
                    ) / tar_norm
                    loss = raw_loss.mean() / self.params.accum_grad
                    forward_end = time.time()
                    forward_time = forward_end - model_start
                    with paddle.no_grad():
                        logs["train_l1"] += paddle.nn.functional.l1_loss(
                            input=output, label=tar
                        )
                        log_nrmse = raw_loss.sqrt().mean()
                        last_log_nrmse = float(log_nrmse.item())
                        logs["train_nrmse"] += log_nrmse
                        loss_logs[dset_type] += loss.item()
                        logs["train_rmse"] += (
                            residuals.pow(2).mean(spatial_dims).sqrt().mean()
                        )
                    self.gscaler.scale(loss).backward()
                    backward_end = time.time()
                    backward_time = backward_end - forward_end
                    optimizer_step = 0
                    if should_sync:
                        self.gscaler.unscale_(self.optimizer)
                        paddle.nn.utils.clip_grad_norm_(
                            parameters=self.model.parameters(), max_norm=1
                        )
                        self.gscaler.step(self.optimizer)
                        self.gscaler.update()
                        self.optimizer.clear_grad()
                        if self.scheduler is not None:
                            self.scheduler.step()
                        optimizer_step = time.time() - backward_end
                    tr_time += time.time() - model_start
                    bench_fwd.append(forward_time)
                    bench_bwd.append(backward_time)
                    bench_opt.append(optimizer_step or 0.0)
                    bench_step.append(forward_time + backward_time + (optimizer_step or 0.0))
                    bench_loss.append(last_log_nrmse)
                    if (
                        self.log_to_screen
                        and batch_idx % self.params.log_interval == 0
                        and self.global_rank == 0
                    ):
                        print(
                            f"Epoch {self.epoch} Batch {batch_idx} Train Loss {log_nrmse.item()}"
                        )
                    if self.log_to_screen:
                        print(
                            "Total Times. Batch: {}, Rank: {}, Data Shape: {}, Data time: {}, Forward: {}, Backward: {}, Optimizer: {}".format(
                                batch_idx,
                                self.global_rank,
                                inp.shape,
                                dtime,
                                forward_time,
                                backward_time,
                                optimizer_step,
                            )
                        )
                    data_start = time.time()
        if self.log_to_screen and bench_fwd:
            warm = min(int(getattr(self.params, "bench_warmup", 5)), max(1, len(bench_fwd) - 1))

            def _stat(xs):
                a = np.asarray(xs[warm:], dtype=np.float64)
                if a.size == 0:
                    return dict(n=0)
                return dict(
                    n=a.size,
                    mean=float(a.mean()),
                    median=float(np.median(a)),
                    std=float(a.std()),
                    mn=float(a.min()),
                    mx=float(a.max()),
                    total=float(a.sum()),
                )

            sf, sb, so, ss = (
                _stat(bench_fwd),
                _stat(bench_bwd),
                _stat(bench_opt),
                _stat(bench_step),
            )
            self.single_print(
                "[BENCH] measured={} warmup={} | fwd med={:.4f} mean={:.4f} std={:.4f} "
                "| bwd med={:.4f} mean={:.4f} | opt med={:.4f} | step med={:.4f} mean={:.4f} std={:.4f} "
                "| loss0={:.6f} lossN={:.6f}".format(
                    sf["n"],
                    warm,
                    sf["median"],
                    sf["mean"],
                    sf["std"],
                    sb["median"],
                    sb["mean"],
                    so["median"],
                    ss["median"],
                    ss["mean"],
                    ss["std"],
                    bench_loss[warm] if len(bench_loss) > warm else float("nan"),
                    bench_loss[-1] if bench_loss else float("nan"),
                )
            )
            # 落盘原始逐步计时，便于离线统计分析
            try:
                os.makedirs(self.params.experiment_dir, exist_ok=True)
                np.savez(
                    os.path.join(self.params.experiment_dir, "bench_timings.npz"),
                    fwd=np.asarray(bench_fwd),
                    bwd=np.asarray(bench_bwd),
                    opt=np.asarray(bench_opt),
                    step=np.asarray(bench_step),
                    loss=np.asarray(bench_loss),
                    warmup=warm,
                )
                self.single_print(
                    f"[BENCH] saved bench_timings.npz to {self.params.experiment_dir}"
                )
            except Exception as e:
                self.single_print(f"[BENCH] failed to save timings: {e}")
        logs = {k: (v / steps) for k, v in logs.items()}
        if paddle.distributed.is_initialized():
            for key in sorted(logs.keys()):
                paddle.distributed.all_reduce(tensor=logs[key].detach())
                logs[key] = float(logs[key] / paddle.distributed.get_world_size())
            for key in sorted(loss_logs.keys()):
                paddle.distributed.all_reduce(tensor=loss_logs[key].detach())
            for key in sorted(grad_logs.keys()):
                paddle.distributed.all_reduce(tensor=grad_logs[key].detach())
            for key in sorted(loss_counts.keys()):
                paddle.distributed.all_reduce(tensor=loss_counts[key].detach())
            for key in sorted(grad_counts.keys()):
                paddle.distributed.all_reduce(tensor=grad_counts[key].detach())
        for key in loss_logs.keys():
            logs[f"{key}/train_nrmse"] = loss_logs[key] / loss_counts[key]
        self.iters += steps
        if self.global_rank == 0:
            logs["iters"] = self.iters
        self.single_print("all reduces executed!")
        return tr_time, data_time, logs

    def validate_one_epoch(self, full=False):
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD.
        """
        if self.params.use_ddp and paddle.distributed.is_initialized():
            paddle.distributed.barrier()
            if self.global_rank != 0:
                paddle.distributed.barrier()
                return {}
        self.model.eval()
        if full:
            cutoff = 999999999999
        else:
            cutoff = 40
        self.single_print("STARTING VALIDATION!!!")
        with paddle.no_grad():
            with paddle.amp.auto_cast(enable=False, dtype=self.mp_type):
                field_labels = self.valid_dataset.get_state_names()
                distinct_dsets = list(
                    set(
                        [
                            dset.title
                            for dset_group in self.valid_dataset.sub_dsets
                            for dset in dset_group.get_per_file_dsets()
                        ]
                    )
                )
                counts = {dset: (0) for dset in distinct_dsets}
                logs = {}
                for subset_group in self.valid_dataset.sub_dsets:
                    for subset in subset_group.get_per_file_dsets():
                        dset_type = subset.title
                        self.single_print("VALIDATING ON", dset_type)
                        temp_loader = paddle.io.DataLoader(
                            dataset=subset,
                            batch_size=self.params.batch_size,
                            num_workers=self.params.num_data_workers,
                            shuffle=not self.params.use_ddp,
                            drop_last=True,
                        )
                        count = 0
                        for batch_idx, data in enumerate(temp_loader):
                            if count > cutoff:
                                del temp_loader
                                break
                            count += 1
                            counts[dset_type] += 1
                            inp, bcs, tar = map(lambda x: x.to(self.device), data)
                            labels = (
                                paddle.to_tensor(
                                    self.train_dataset.subset_dict.get(
                                        subset.get_name(),
                                        [-1]
                                        * len(
                                            self.valid_dataset.subset_dict[
                                                subset.get_name()
                                            ]
                                        ),
                                    ),
                                )
                                .to(self.device)
                                .unsqueeze(0)
                                .expand(tar.shape[0], -1)
                            )
                            inp = einops.rearrange(inp, "b t c h w -> t b c h w")
                            output = self.model(inp, labels, bcs)
                            spatial_dims = tuple(range(output.ndim))[2:]
                            residuals = output - tar
                            nmse = (
                                residuals.pow(2).mean(spatial_dims, keepdim=True)
                                / (1e-07 + tar.pow(2).mean(spatial_dims, keepdim=True))
                            ).sqrt()
                            logs[f"{dset_type}/valid_nrmse"] = (
                                logs.get(f"{dset_type}/valid_nrmse", 0) + nmse.mean()
                            )
                            logs[f"{dset_type}/valid_rmse"] = (
                                logs.get(f"{dset_type}/valid_mse", 0)
                                + residuals.pow(2).mean(spatial_dims).sqrt().mean()
                            )
                            logs[f"{dset_type}/valid_l1"] = (
                                logs.get(f"{dset_type}/valid_l1", 0)
                                + residuals.abs().mean()
                            )
                            for i, field in enumerate(
                                self.valid_dataset.subset_dict[subset.type]
                            ):
                                field_name = field_labels[field]
                                logs[f"{dset_type}/{field_name}_valid_nrmse"] = (
                                    logs.get(f"{dset_type}/{field_name}_valid_nrmse", 0)
                                    + nmse[:, i].mean()
                                )
                                logs[f"{dset_type}/{field_name}_valid_rmse"] = (
                                    logs.get(f"{dset_type}/{field_name}_valid_rmse", 0)
                                    + residuals[:, i : i + 1]
                                    .pow(2)
                                    .mean(spatial_dims)
                                    .sqrt()
                                    .mean()
                                )
                                logs[f"{dset_type}/{field_name}_valid_l1"] = (
                                    logs.get(f"{dset_type}/{field_name}_valid_l1", 0)
                                    + residuals[:, i].abs().mean()
                                )
                        else:
                            del temp_loader
            self.single_print("DONE VALIDATING - NOW SYNCING")
            for k, v in logs.items():
                dset_type = k.split("/")[0]
                logs[k] = v / counts[dset_type]
            logs["valid_nrmse"] = 0
            for dset_type in distinct_dsets:
                logs["valid_nrmse"] += logs[f"{dset_type}/valid_nrmse"] / len(
                    distinct_dsets
                )
            if paddle.distributed.is_initialized():
                for key in sorted(logs.keys()):
                    paddle.distributed.all_reduce(tensor=logs[key].detach())
                    logs[key] = float(
                        logs[key].item() / paddle.distributed.get_world_size()
                    )
                    if "rmse" in key:
                        logs[key] = logs[key]
            self.single_print("DONE SYNCING - NOW LOGGING")
        if self.params.use_ddp and paddle.distributed.is_initialized():
            paddle.distributed.barrier()
        return logs

    def train(self):
        if self.params.log_to_wandb:
            if self.sweep_id:
                wandb.init(dir=self.params.experiment_dir)
                hpo_config = wandb.config.as_dict()
                self.params.update_params(hpo_config)
                params = self.params
            else:
                wandb.init(
                    dir=self.params.experiment_dir,
                    config=self.params,
                    name=self.params.name,
                    group=self.params.group,
                    project=self.params.project,
                    entity=self.params.entity,
                    resume=True,
                )
        if self.sweep_id and paddle.distributed.is_initialized():
            param_file = f"temp_hpo_config_{os.environ['SLURM_JOBID']}.pkl"
            if self.global_rank == 0:
                with open(param_file, "wb") as f:
                    pkl.dump(hpo_config, f)
            paddle.distributed.barrier()
            if self.global_rank != 0:
                with open(param_file, "rb") as f:
                    hpo_config = pkl.load(f)
            paddle.distributed.barrier()
            if self.global_rank == 0:
                os.remove(param_file)
            if "batch_size" in hpo_config:
                hpo_config["batch_size"] = int(
                    hpo_config["batch_size"] // self.world_size
                )
            self.params.update_params(hpo_config)
            params = self.params
            self.initialize_data(self.params)
            self.initialize_model(self.params)
            self.initialize_optimizer(self.params)
            self.initialize_scheduler(self.params)
        # if self.global_rank == 0:
        # summary(self.model)
        if self.params.log_to_wandb:
            wandb.watch(self.model)
        self.single_print("Starting Training Loop...")
        best_valid_loss = 1000000.0
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if paddle.distributed.is_initialized():
                self.train_sampler.set_epoch(epoch)
            start = time.time()
            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_start = time.time()
            if getattr(self.params, "bench_mode", False):
                valid_logs = {"valid_nrmse": 0.0}
            elif epoch == self.params.max_epochs - 1:
                valid_logs = self.validate_one_epoch(True)
            else:
                valid_logs = self.validate_one_epoch()
            post_start = time.time()
            train_logs.update(valid_logs)
            train_logs["time/train_time"] = valid_start - start
            train_logs["time/train_data_time"] = data_time
            train_logs["time/train_compute_time"] = tr_time
            train_logs["time/valid_time"] = post_start - valid_start
            if self.params.log_to_wandb:
                wandb.log(train_logs)
            gc.collect()
            paddle.device.cuda.empty_cache()
            if self.global_rank == 0:
                if not getattr(self.params, "bench_mode", False):
                    if self.params.save_checkpoint:
                        self.save_checkpoint(self.params.checkpoint_path)
                    if epoch % self.params.checkpoint_save_interval == 0:
                        self.save_checkpoint(self.params.checkpoint_path + f"_epoch{epoch}")
                    if valid_logs["valid_nrmse"] <= best_valid_loss:
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs["valid_nrmse"]
                cur_time = time.time()
                self.single_print(
                    f"Time for train {valid_start - start}. For valid: {post_start - valid_start}. For postprocessing:{cur_time - post_start}"
                )
                self.single_print(
                    "Time taken for epoch {} is {} sec".format(
                        epoch + 1, time.time() - start
                    )
                )
                self.single_print(
                    "Train loss: {}. Valid loss: {}".format(
                        train_logs["train_nrmse"], valid_logs["valid_nrmse"]
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="00", type=str)
    parser.add_argument(
        "--use_ddp", action="store_true", help="Use distributed data parallel"
    )
    parser.add_argument("--yaml_config", default="./config/multi_ds.yaml", type=str)
    parser.add_argument("--config", default="basic_config", type=str)
    parser.add_argument(
        "--sweep_id",
        default=None,
        type=str,
        help="sweep config from ./configs/sweeps.yaml",
    )
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = args.use_ddp
    has_cuda_device = (
        paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    )
    global_rank = 0
    local_rank = 0
    world_size = 1
    if args.use_ddp:
        paddle.distributed.init_parallel_env()
        parallel_env = paddle.distributed.ParallelEnv()
        global_rank = int(paddle.distributed.get_rank())
        local_rank = int(parallel_env.local_rank)
        world_size = int(paddle.distributed.get_world_size())
        if has_cuda_device:
            paddle.device.set_device(f"gpu:{local_rank}")
    elif has_cuda_device:
        paddle.device.set_device("gpu:0")
    device = f"gpu:{local_rank}" if has_cuda_device else "cpu"
    if params.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch_size ({params.batch_size}) must be divisible by world_size ({world_size})."
        )
    params["batch_size"] = int(params.batch_size // world_size)
    if params.batch_size < 1:
        raise ValueError(
            "Per-rank batch_size became 0 after sharding. Increase the configured global batch_size."
        )
    params["startEpoch"] = 0
    if args.sweep_id:
        jid = os.environ["SLURM_JOBID"]
        expDir = os.path.join(
            params.exp_dir, args.sweep_id, args.config, str(args.run_name), jid
        )
    else:
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))
    params["old_exp_dir"] = expDir
    params["experiment_dir"] = os.path.abspath(expDir)
    params["checkpoint_path"] = os.path.join(expDir, "training_checkpoints/ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/best_ckpt.tar"
    )
    params["old_checkpoint_path"] = os.path.join(
        params.old_exp_dir, "training_checkpoints/best_ckpt.tar"
    )
    if global_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, "training_checkpoints/"))
    params["resuming"] = True if os.path.isfile(params.checkpoint_path) else False
    params["name"] = str(args.run_name)
    if global_rank == 0:
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        logging_utils.log_versions()
        params.log()
    params["log_to_wandb"] = global_rank == 0 and params["log_to_wandb"]
    params["log_to_screen"] = global_rank == 0 and params["log_to_screen"]
    PaddleFlag.cudnn_benchmark = False
    if global_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, "hyperparams.yaml"), "w") as hpfile:
            yaml.dump(hparams, hpfile)
    trainer = Trainer(params, global_rank, local_rank, device, sweep_id=args.sweep_id)
    if args.sweep_id and trainer.global_rank == 0:
        print(args.sweep_id, trainer.params.entity, trainer.params.project)
        wandb.agent(
            args.sweep_id,
            function=trainer.train,
            count=1,
            entity=trainer.params.entity,
            project=trainer.params.project,
        )
    else:
        trainer.train()
    if params.log_to_screen:
        print("DONE ---- rank %d" % global_rank)
