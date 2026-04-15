import os
from logging import getLogger

import numpy as np
import paddle

from ppcfd.models.prose_fd.runtime import get_amp_device_type, max_memory_allocated_mb, to_cuda

try:
    from examples.prose_fd.data_utils.collate import custom_collate
    from examples.prose_fd.dataset import get_dataset
    from examples.prose_fd.utils.dadapt_adan_paddle import DAdaptAdan
    from examples.prose_fd.utils.lr_scheduler import build_lr_scheduler
except ImportError:
    from data_utils.collate import custom_collate
    from dataset import get_dataset
    from utils.dadapt_adan_paddle import DAdaptAdan
    from utils.lr_scheduler import build_lr_scheduler

logger = getLogger()


class Trainer(object):
    def __init__(self, modules, params, symbol_env):
        """
        Initialize trainer.
        """
        self.modules = modules
        self.params = params
        self.symbol_env = symbol_env
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = 0
        self.set_parameters()
        if params.multi_gpu:
            logger.info("Using paddle.DataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = paddle.DataParallel(self.modules[k])
        self.set_optimizer()
        self.scaler = None
        if params.amp:
            self.scaler = paddle.amp.GradScaler(enable=True)
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-np.infty if biggest else np.infty)
            for metric, biggest in self.metrics
        }
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.reload_checkpoint()
        if not params.eval_only:
            self.dataloader_count = 0
            self.dataset = get_dataset(params, symbol_env, split="train")
            self.dataloader = paddle.io.DataLoader(
                dataset=self.dataset,
                batch_size=params.batch_size,
                num_workers=params.num_workers,
                drop_last=True,
                collate_fn=custom_collate(
                    params.data.max_output_dimension,
                    symbol_env.pad_index,
                    params.data.tie_fields,
                    self.params.data.get("mixed_length", 0),
                    params.input_len,
                    params.symbol.pad_right,
                ),
            )
            self.data_iter = iter(self.dataloader)
        self.data_loss = 0.0
        if not params.use_raw_time:
            self.input_len = params.input_len
            self.output_len = params.data.t_num - self.input_len
            if params.rollout:
                self.t = paddle.linspace(
                    0, 10, self.input_len + 1, dtype=paddle.float32
                )[None]
            else:
                self.t = paddle.linspace(
                    0, 10, params.data.t_num, dtype=paddle.float32
                )[None]

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if not p.stop_gradient]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            num = sum([p.numel() for p in v])
            logger.info(f"Found {num:,} parameters in {k}.")
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.scheduler = None
        optimizer_lr = params.optim.lr if params.optim.type == "adamw" else 1.0
        if params.optim.scheduler_type:
            if params.optim.scheduler_type == "one_cycle":
                self.scheduler = paddle.optimizer.lr.OneCycleLR(
                    max_learning_rate=params.optim.lr,
                    divide_factor=10000.0,
                    phase_pct=params.optim.warmup / params.optim.max_iters,
                    total_steps=params.n_steps_per_epoch * params.max_epoch,
                    end_learning_rate=params.optim.lr * 1.0 / (10000.0 * 10000.0),
                )
            else:
                scheduler_args = {}
                if params.optim.scheduler_type == "cosine_with_restarts":
                    scheduler_args["num_cycles"] = params.optim.get("num_cycles", 1)
                elif params.optim.scheduler_type == "cosine_with_min_lr":
                    if "min_lr" in params.optim:
                        scheduler_args["min_lr"] = params.optim.min_lr
                    if "min_lr_rate" in params.optim:
                        scheduler_args["min_lr_rate"] = params.optim.min_lr_rate
                elif params.optim.scheduler_type == "warmup_stable_decay":
                    scheduler_args["num_decay_steps"] = int(
                        params.optim.max_iters * params.optim.decay
                    )
                    scheduler_args["min_lr_ratio"] = params.optim.get(
                        "min_lr_ratio", 0
                    )
                    scheduler_args["num_stable_steps"] = (
                        params.optim.max_iters
                        - params.optim.warmup
                        - scheduler_args["num_decay_steps"]
                    )
                self.scheduler = build_lr_scheduler(
                    scheduler_type=params.optim.scheduler_type,
                    base_learning_rate=optimizer_lr,
                    num_warmup_steps=params.optim.warmup,
                    num_training_steps=params.optim.max_iters,
                    scheduler_specific_kwargs=scheduler_args,
                )
        learning_rate = self.scheduler if self.scheduler is not None else optimizer_lr
        if params.optim.type == "adamw":
            self.optimizer = paddle.optimizer.AdamW(
                learning_rate=learning_rate,
                parameters=self.parameters["model"],
                weight_decay=params.optim.weight_decay,
                epsilon=params.optim.get("eps", 1e-08),
                amsgrad=params.optim.get("amsgrad", False),
                beta1=0.9,
                beta2=params.optim.get("beta2", 0.999),
            )
        elif params.optim.type == "adan":
            self.optimizer = DAdaptAdan(
                self.parameters["model"],
                lr=learning_rate,
                weight_decay=params.optim.weight_decay,
                growth_rate=1.05,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {params.optim.type}")
        logger.info(
            f"Optimizer: {type(self.optimizer)}, scheduler: {type(self.scheduler)}"
        )

    def optimize(self, loss):
        """
        Optimize.
        """
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            exit()
        params = self.params
        optimizer = self.optimizer
        if params.accumulate_gradients > 1:
            loss = loss / params.accumulate_gradients
        if not params.amp:
            loss.backward()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    paddle.nn.utils.clip_grad_norm_(
                        parameters=self.parameters["model"],
                        max_norm=params.clip_grad_norm,
                    )
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer.zero_grad()
        else:
            self.scaler.scale(loss).backward()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    paddle.nn.utils.clip_grad_norm_(
                        parameters=self.parameters["model"],
                        max_norm=params.clip_grad_norm,
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                optimizer.zero_grad()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.params.print_freq != 0:
            return
        s_iter = "%7i - " % self.n_total_iter
        s_lr = f" - LR: {self.optimizer.get_lr():.4e}"
        max_mem = max_memory_allocated_mb()
        s_mem = "" if max_mem is None else " MEM: {:.2f} MB - ".format(max_mem)
        logger.info(s_iter + s_mem + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return
        path = os.path.join(self.params.dump_path, f"{name}.pth")
        logger.info(f"Saving {name} to {path} ...")
        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "dataloader_count": self.dataloader_count,
            "best_metrics": self.best_metrics,
        }
        for k, v in self.modules.items():
            data[k] = v.state_dict()
        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                scaler_state = self.scaler.state_dict()
                if scaler_state:
                    data["scaler"] = scaler_state
                else:
                    logger.info("Skipping empty gradient scaler state when saving checkpoint.")
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
            logger.warning(f"Saving model and optimizer parameters ...")
        else:
            logger.warning(f"Saving model parameters ...")
        paddle.save(obj=data, path=path)

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"
        if self.params.reload_checkpoint is not None:
            checkpoint_path = os.path.join(self.params.reload_checkpoint, path)
            assert os.path.isfile(checkpoint_path)
        else:
            if root is not None:
                checkpoint_path = os.path.join(root, path)
            else:
                checkpoint_path = os.path.join(self.params.dump_path, path)
            if not os.path.isfile(checkpoint_path):
                logger.warning(
                    "Checkpoint path does not exist, {}".format(checkpoint_path)
                )
                return
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = paddle.load(path=str(checkpoint_path))
        for k, v in self.modules.items():
            try:
                weights = data[k]
                v.load_state_dict(weights)
            except RuntimeError:
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            v.stop_gradient = not requires_grad
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])
        scaler_state = data.get("scaler")
        if self.scaler is not None and scaler_state:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(scaler_state)
        elif self.scaler is not None and "scaler" in data:
            logger.warning("Skipping empty gradient scaler state from checkpoint.")
        if "scheduler" in data and self.scheduler is not None:
            logger.warning("Reloading scheduler...")
            self.scheduler.set_state_dict(data["scheduler"])
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.dataloader_count = data["dataloader_count"]
        self.best_metrics = data["best_metrics"]
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch > 0
            and self.epoch % self.params.save_periodic == 0
        ):
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores, prefix=None, suffix=None):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            _metric = metric
            if prefix is not None:
                _metric = prefix + "_" + _metric
            if suffix is not None:
                _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1
            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[_metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self):
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self):
        """
        Return a training batch
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.dataloader_count += 1
            logger.info(
                f"Reached end of dataloader, restart {self.dataloader_count}..."
            )
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

    def data_loss_fn(self, data_output, data_label, data_mask):
        """
        data_output/data_label: Tensor (bs, output_len, x_num, x_num, dim)
        """
        if self.params.loss_weight == "l2":
            weight = paddle.linalg.vector_norm(data_label, dim=(2, 3), keepdim=True)
        elif self.params.loss_weight == "linfty":
            weight, _ = paddle.compat.max(
                paddle.abs(data_label), dim=(2, 3), keepdim=True
            )
        else:
            weight = None
        if weight is None:
            loss = paddle.nn.functional.mse_loss(
                input=data_output, label=data_label, reduction="none"
            )
            loss = loss.sum() / paddle.count_nonzero(x=data_mask.expand_as(loss))
        else:
            eps = 1e-06
            if self.params.square_loss:
                loss = paddle.nn.functional.mse_loss(
                    input=data_output, label=data_label, reduction="none"
                )
                loss = (loss / (weight**2 + eps)).sum() / data_label.size(0)
            else:
                loss = paddle.linalg.vector_norm(
                    data_output - data_label, dim=(2, 3), keepdim=True
                )
                loss = (loss / (weight + eps)).sum() / data_label.size(0)
        return loss

    def normalize_data(self, data_input, data_label=None):
        if self.params.normalize:
            eps = 1e-08
            if self.params.normalize == "meanvar":
                mean = paddle.mean(data_input, axis=(1, 2, 3), keepdim=True)
                std = paddle.std(data_input, axis=(1, 2, 3), keepdim=True) + eps
            elif self.params.normalize == "range":
                max = paddle.amax(data_input, dim=(1, 2, 3), keepdim=True)
                min = paddle.amin(data_input, dim=(1, 2, 3), keepdim=True)
                mean = (max + min) / 2
                std = (max - min) / 2 + eps
            elif self.params.normalize == "meanvar_c":
                mean = paddle.mean(data_input, axis=(1, 2, 3, 4), keepdim=True)
                std = paddle.std(data_input, axis=(1, 2, 3, 4), keepdim=True) + eps
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.params.normalize}"
                )
            data_input = (data_input - mean) / std
            if not self.params.denormalize_for_loss and data_label is not None:
                data_label = (data_label - mean) / std
        else:
            mean = 0
            std = 1
        return data_input, data_label, mean, std

    def prepare_data(self, samples, train=True):
        """
        Prepare data for training. (Split entire sequence into input and output, generate loss mask, move to cuda, etc.)

        samples: data:         Tensor     (bs, max_len, x_num, x_num, dim)
                 data_mask:    BoolTensor (bs, 1/output_len, 1, 1, dim)
                 t:            Tensor     (bs, max_len)

        """
        model_input = {}
        data = samples["data"]
        data_mask = samples["data_mask"]
        if self.params.use_raw_time:
            t = samples["t"]
        else:
            t = self.t
        input_len = self.params.input_len
        data_input = data[:, :input_len]
        data_label = data[:, input_len:]
        data_input, data_label, data_mask = to_cuda(data_input, data_label, data_mask)
        data_input, data_label, mean, std = self.normalize_data(data_input, data_label)
        input_times = t[:, :input_len, None]
        output_times = t[:, input_len:, None] - input_times[:, -1:]
        model_input["input_times"] = to_cuda(input_times)
        model_input["output_times"] = to_cuda(output_times)
        model_input["data_input"] = data_input
        d = {"data_label": data_label, "data_mask": data_mask, "mean": mean, "std": std}
        if "symbol_input" in samples:
            model_input["symbol_input"] = to_cuda(samples["symbol_input"])
            model_input["symbol_padding_mask"] = to_cuda(samples["symbol_mask"])
        return model_input, d

    def iter(self):
        """
        One training step.
        """
        params = self.params
        samples = self.get_batch()
        model = self.modules["model"]
        model.train()
        model_input, d = self.prepare_data(samples)
        """
        Model input: 
            check prepare_data() function

        Model output:
            data_output:  (bs, output_len, x_num, x_num, data_dim)
        """
        with paddle.amp.autocast(
            get_amp_device_type(),
            enabled=bool(params.amp),
            dtype=paddle.bfloat16,
        ):
            data_output = model("fwd", **model_input)
            if self.params.normalize and self.params.denormalize_for_loss:
                data_output = data_output * d["std"] + d["mean"]
            data_output = data_output * d["data_mask"]
            data_loss = self.data_loss_fn(data_output, d["data_label"], d["data_mask"])
        self.data_loss += data_loss.item()
        self.optimize(data_loss)
        self.inner_epoch += 1
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()
