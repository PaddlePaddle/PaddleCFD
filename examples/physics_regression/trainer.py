import copy
import io
import json
import os
import pickle
import sys
import time
from collections import OrderedDict, defaultdict
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import paddle
import pandas as pd
import seaborn as sns

from optim import get_optimizer
from ppcfd.models.physicsregression.paddle_utils import *
from ppcfd.models.physicsregression.symbolicregression.utils import to_cuda

has_apex = True
try:
    import apex
except:
    has_apex = False
logger = getLogger()


class LoadParameters(object):
    def __init__(self, modules, params):
        self.modules = modules
        self.params = params
        self.set_parameters()

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"
        if root is None:
            root = self.params.dump_path
        checkpoint_path = os.path.join(root, path)
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint + "/checkpoint.pth"
                assert os.path.isfile(checkpoint_path)
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


class Trainer(object):
    def __init__(self, modules, env, params, path=None, root=None):
        """
        Initialize trainer.
        """
        self.modules = modules
        self.params = params
        self.env = env
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = self.total_samples = self.n_equations = 0
        self.infos_statistics = defaultdict(list)
        self.errors_statistics = defaultdict(int)
        self.iterators = {}
        self.set_parameters()
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        assert not params.nvidia_apex or has_apex
        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = paddle.DataParallel(layers=self.modules[k])
        self.set_optimizer()
        self.scaler = None
        if params.amp >= 0:
            self.init_amp()
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = split[0][1:], False
            else:
                self.stopping_criterion = split[0], True
            self.best_stopping_criterion = (
                -1000000000000.0 if self.stopping_criterion[1] else 1000000000000.0
            )
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-np.inf if biggest else np.inf)
            for metric, biggest in self.metrics
        }
        self.epoch = 1
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [("processed_e", 0)]
            + [("processed_w", 0)]
            + sum(
                [[(x, []), (f"{x}-AVG-STOP-PROBS", [])] for x in env.TRAINING_TASKS], []
            )
        )
        self.last_time = time.time()
        self.reload_checkpoint(path=path, root=root)
        if params.export_data:
            assert params.reload_data == ""
            params.export_path_prefix = os.path.join(params.dump_path, "data.prefix")
            self.file_handler_prefix = io.open(
                params.export_path_prefix, mode="a", encoding="utf-8"
            )
            logger.info(
                f"Data will be stored in prefix in: {params.export_path_prefix} ..."
            )
        if params.reload_data != "":
            logger.info(params.reload_data)
            assert params.export_data is False
            s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
            assert len(s) >= 1
            self.data_path = {
                task: (
                    train_path if train_path != "" else None,
                    valid_path if valid_path != "" else None,
                    test_path if test_path != "" else None,
                )
                for task, train_path, valid_path, test_path in s
            }
            logger.info(self.data_path)
            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1000000000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, self.data_path, params))
                for task in params.tasks
            }

    def set_new_train_iterator_params(self, args={}):
        params = self.params
        if params.env_base_seed < 0:
            params.env_base_seed = np.random.randint(1000000000)
        self.dataloader = {
            task: iter(
                self.env.create_train_iterator(task, self.data_path, params, args)
            )
            for task in params.tasks
        }
        logger.info(
            "Succesfully replaced training iterator with following args:{}".format(args)
        )
        return

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(
            self.parameters["model"], params.lr, params.optimizer
        )
        logger.info("Optimizer: %s" % type(self.optimizer))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        mod_names = sorted(self.modules.keys())
        if params.nvidia_apex is True:
            modules, optimizer = apex.amp.initialize(
                [self.modules[k] for k in mod_names],
                self.optimizer,
                opt_level="O%i" % params.amp,
            )
            self.modules = {k: module for k, module in zip(mod_names, modules)}
            self.optimizer = optimizer
        else:
            self.scaler = paddle.amp.GradScaler(
                incr_every_n_steps=2000, init_loss_scaling=65536.0
            )

    def optimize(self, loss):
        """
        Optimize.
        """
        if (loss != loss).data.any():
            logger.warning("NaN detected")
        params = self.params
        optimizer = self.optimizer
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                paddle.nn.utils.clip_grad_norm_(
                    parameters=self.parameters["model"], max_norm=params.clip_grad_norm
                )
            optimizer.step()
        elif params.nvidia_apex is True:
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    paddle.nn.utils.clip_grad_norm_(
                        parameters=apex.amp.master_params(self.optimizer),
                        max_norm=params.clip_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(
                    loss, optimizer, delay_unscale=True
                ) as scaled_loss:
                    scaled_loss.backward()
        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
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
                optimizer.zero_grad()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.params.print_freq != 0:
            return
        s_total_eq = "- Total Eq: " + "{:.2e}".format(self.n_equations)
        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]
        # 增强学习率日志记录
        current_lr = self.optimizer.get_lr()
        s_lr = " - LR: {:.4e}".format(current_lr)

        # 添加调度器状态信息（如果可用）
        if hasattr(self.optimizer, 'num_updates'):
            s_lr += " (updates: {})".format(self.optimizer.num_updates)

            # 显示是否在warmup阶段
            if hasattr(self.optimizer, 'warmup_updates'):
                if self.optimizer.num_updates < self.optimizer.warmup_updates:
                    warmup_progress = self.optimizer.num_updates / self.optimizer.warmup_updates * 100
                    s_lr += " [WARMUP: {:.1f}%]".format(warmup_progress)
                else:
                    s_lr += " [POST-WARMUP]"
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        max_mem = paddle.cuda.max_memory_allocated() / 1024**2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)
        self.stats["processed_e"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time
        logger.info(s_iter + s_speed + s_mem + s_stat + s_lr + s_total_eq)

    def get_generation_statistics(self, task):
        total_eqs = sum(
            x.shape[0]
            for x in self.infos_statistics[list(self.infos_statistics.keys())[0]]
        )
        logger.info("Generation statistics (to generate {} eqs):".format(total_eqs))
        all_infos = defaultdict(list)
        for info_type, infos in self.infos_statistics.items():
            all_infos[info_type] = paddle.cat(infos).tolist()
            infos = [paddle.bincount(x=info) for info in infos]
            max_val = max([info.shape[0] for info in infos])
            aggregated_infos = paddle.cat(
                [
                    paddle.compat.nn.functional.pad(
                        info, (0, max_val - info.shape[0])
                    ).unsqueeze(-1)
                    for info in infos
                ],
                -1,
            ).sum(-1)
            non_zeros = aggregated_infos.nonzero(as_tuple=True)[0]
            vals = [
                (
                    non_zero.item(),
                    "{:.2e}".format(
                        (aggregated_infos[non_zero] / aggregated_infos.sum()).item()
                    ),
                )
                for non_zero in non_zeros
            ]
            logger.info("{}: {}".format(info_type, vals))
        all_infos = pd.DataFrame(all_infos)
        g = sns.PairGrid(all_infos)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
        plt.savefig(
            os.path.join(self.params.dump_path, "statistics_{}.png".format(self.epoch))
        )
        str_errors = "Errors ({} eqs)\n ".format(total_eqs)
        for error_type, count in self.errors_statistics.items():
            str_errors += "{}: {}, ".format(error_type, count)
        logger.info(str_errors[:-2])
        self.errors_statistics = defaultdict(int)
        self.infos_statistics = defaultdict(list)
        # 清理未使用的loss累积器（如果存在）
        if hasattr(self, '_loss_accumulator'):
            del self._loss_accumulator

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return
        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))
        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }
        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()
        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()

            # 新增：保存学习率调度器状态
            if hasattr(self.optimizer, 'num_updates'):
                data["optimizer_num_updates"] = self.optimizer.num_updates
                logger.warning(f"Saving optimizer num_updates: {self.optimizer.num_updates}")

            # 保存当前学习率（用于验证）
            if hasattr(self.optimizer, '_learning_rate'):
                data["optimizer_current_lr"] = self.optimizer._learning_rate
                logger.warning(f"Saving current learning rate: {self.optimizer._learning_rate}")

            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()
        paddle.save(obj=data, path=path)

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"
        if os.path.isfile(self.params.reload_checkpoint):
            checkpoint_path = self.params.reload_checkpoint
        elif self.params.reload_checkpoint != "":
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
        if "checkpoint" in checkpoint_path:
            for k, v in self.modules.items():
                weights = data[k]
                try:
                    weights = data[k]
                    v.load_state_dict(weights)
                except RuntimeError:
                    weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                    v.load_state_dict(weights)
                v.stop_gradient = not requires_grad
            if self.params.amp == -1 or not self.params.nvidia_apex:
                logger.warning("Reloading checkpoint optimizer ...")
                self.optimizer.load_state_dict(data["optimizer"])

                # 新增：恢复学习率调度器状态
                if hasattr(self.optimizer, 'num_updates') and "optimizer_num_updates" in data:
                    self.optimizer.num_updates = data["optimizer_num_updates"]
                    logger.warning(f"Restored optimizer num_updates: {self.optimizer.num_updates}")

                    # 重新计算并设置学习率
                    if hasattr(self.optimizer, 'get_lr_for_step'):
                        restored_lr = self.optimizer.get_lr_for_step(self.optimizer.num_updates)
                        self.optimizer._learning_rate = restored_lr
                        logger.warning(f"Restored learning rate: {restored_lr}")
                else:
                    if "optimizer_num_updates" not in data:
                        logger.warning("Old checkpoint format detected. Learning rate scheduling will restart.")
                        logger.warning("Consider retraining from a newer checkpoint for optimal performance.")
                    else:
                        logger.warning("No num_updates found in optimizer or optimizer doesn't support it")
            else:
                logger.warning("Not reloading checkpoint optimizer.")
                for group_id, param_group in enumerate(self.optimizer._param_groups):
                    if "num_updates" not in param_group:
                        logger.warning("No 'num_updates' for optimizer.")
                        continue
                    logger.warning("Reloading 'num_updates' and 'lr' for optimizer.")
                    param_group["num_updates"] = data["optimizer"]["_param_groups"][
                        group_id
                    ]["num_updates"]
                    param_group["lr"] = self.optimizer.get_lr_for_step(
                        param_group["num_updates"]
                    )
            if self.params.fp16 and not self.params.nvidia_apex:
                logger.warning("Reloading gradient scaler ...")
                self.scaler.load_state_dict(data["scaler"])
            else:
                assert self.scaler is None and "scaler" not in data
            self.epoch = data["epoch"] + 1
            self.n_total_iter = data["n_total_iter"]
            self.best_metrics = data["best_metrics"]
            self.best_stopping_criterion = data["best_stopping_criterion"]
            logger.warning(
                f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
            )
        else:
            for k, v in self.modules.items():
                weights = data[k]
                try:
                    weights = data[k]
                    v.load_state_dict(weights)
                except RuntimeError:
                    weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                    v.load_state_dict(weights)
                v.stop_gradient = not requires_grad

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch % self.params.save_periodic == 0
            and self.epoch >= self.params.save_periodic_from
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

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        if self.stopping_criterion is not None and (
            self.params.is_master or not self.stopping_criterion[0].endswith("_mt_bleu")
        ):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch, errors = next(self.dataloader[task])
        except Exception as e:
            print(e)
            logger.error(
                "An unknown exception of type {0} occurred in line {1} when fetching batch. Arguments:{2!r}. Restarting ...".format(
                    type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args
                )
            )
            if self.params.is_slurm_job:
                if int(os.environ["SLURM_PROCID"]) == 0:
                    logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
                    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise
        return batch, errors

    def export_data(self, task):
        """
        Export data to the disk.
        """
        samples, _ = self.get_batch(task)
        for info in samples["infos"]:
            samples["infos"][info] = list(map(str, samples["infos"][info].tolist()))

        def get_dictionary_slice(idx, dico):
            x = {}
            for d in dico:
                x[d] = dico[d][idx]
            return x

        def float_list_to_str_lst(lst, float_precision):
            for i in range(len(lst)):
                for j in range(len(lst[i])):
                    str_float = f"%.{float_precision}e" % lst[i][j]
                    lst[i][j] = str_float
            return lst

        processed_e = len(samples)
        for i in range(processed_e):
            outputs = {**get_dictionary_slice(i, samples["infos"])}
            x_to_fit = samples["x_to_fit"][i].tolist()
            y_to_fit = samples["y_to_fit"][i].tolist()
            x_to_predict = samples["x_to_predict"][i].tolist()
            y_to_predict = samples["y_to_predict"][i].tolist()
            outputs["x_to_fit"] = float_list_to_str_lst(
                x_to_fit, self.params.float_precision
            )
            outputs["y_to_fit"] = float_list_to_str_lst(
                y_to_fit, self.params.float_precision
            )
            outputs["x_to_predict"] = float_list_to_str_lst(
                x_to_predict, self.params.float_precision
            )
            outputs["y_to_predict"] = float_list_to_str_lst(
                y_to_predict, self.params.float_precision
            )
            outputs["tree"] = samples["tree"][i].prefix()
            self.file_handler_prefix.write(json.dumps(outputs) + "\n")
            self.file_handler_prefix.flush()
        self.n_equations += self.params.batch_size
        self.total_samples += self.params.batch_size
        self.stats["processed_e"] += len(samples)

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        embedder, encoder, decoder = (
            self.modules["embedder"],
            self.modules["encoder"],
            self.modules["decoder"],
        )
        embedder.train()
        encoder.train()
        decoder.train()
        env = self.env
        if self.inner_epoch == 0:
            self.total_loss = 0
        samples, errors = self.get_batch(task)
        if self.params.debug_train_statistics:
            for info_type, info in samples["infos"].items():
                self.infos_statistics[info_type].append(info)
            for error_type, count in errors.items():
                self.errors_statistics[error_type] += count
        x_to_fit = samples["x_to_fit"]
        y_to_fit = samples["y_to_fit"]
        x1 = []
        for seq_id in range(len(x_to_fit)):
            x1.append([])
            for seq_l in range(len(x_to_fit[seq_id])):
                x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])
        hints = []
        if params.use_hints:
            for used_hints in params.use_hints.split(","):
                hints.append(samples[used_hints])
        if params.use_hints == "units":
            for ii in range(len(hints[0])):
                hints[0][ii] = []
        x1, len1 = embedder(x1, hints)
        if self.params.use_skeleton:
            x2, len2 = self.env.batch_equations(
                self.env.word_to_idx(
                    samples["skeleton_tree_encoded"], float_input=False
                )
            )
            assert False
        else:
            x2, len2, units = self.env.batch_equations(
                self.env.word_to_idx(samples["tree_encoded"], float_input=False),
                units=self.env.word_to_idx(samples["dim_encoded"], unit_input=True),
                decode_physical_units=self.params.decode_physical_units,
            )
        alen = paddle.arange(paddle.max(len2), dtype=paddle.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1
        y = x2[1:].masked_select(pred_mask[:-1])
        if __debug__:  # 只在调试模式下检查，避免频繁GPU-CPU同步
            assert len(y) == (len2 - 1).sum().item()
        if self.params.decode_physical_units == "double-seq":
            y_units = units[1:].masked_select(pred_mask[:-1].unsqueeze(-1))
            units, y_units = to_cuda(units, y_units, device=env.params.device)
        else:
            y_units = None
        x2, len2, y = to_cuda(x2, len2, y, device=env.params.device)
        if params.amp == -1 or params.nvidia_apex:
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
                units=units,
            )
            _, loss = decoder(
                "predict",
                tensor=decoded,
                pred_mask=pred_mask,
                y=y,
                get_scores=False,
                y_units=y_units,
            )
        else:
            with paddle.cuda.amp.autocast():
                encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded.transpose(0, 1),
                    src_len=len1,
                    units=units,
                )
                _, loss = decoder(
                    "predict",
                    tensor=decoded,
                    pred_mask=pred_mask,
                    y=y,
                    get_scores=False,
                    y_units=y_units,
                )
        # 每个batch都统计loss和processed_w，确保统计准确性
        self.stats[task].append(loss.item())
        self.total_loss += loss.item()
        self.optimize(loss)
        self.inner_epoch += 1
        self.n_equations += len1.size(0)
        self.stats["processed_e"] += len1.size(0)
        self.stats["processed_w"] += (len1 + len2 - 2).sum().item()
