import os
from collections import defaultdict
from logging import getLogger

import numpy as np
import paddle
import wandb
from tabulate import tabulate

from ppcfd.models.prose_fd.paddle_utils import *
from ppcfd.models.prose_fd.runtime import get_amp_device_type, sync_tensor, to_cuda

try:
    from examples.prose_fd.data_utils.collate import custom_collate
    from examples.prose_fd.dataset import get_dataset
    from examples.prose_fd.utils.metrics import compute_metrics
    from examples.prose_fd.utils.plot import plot_2d_pde, plot_2d_pde_formal
except ImportError:
    from data_utils.collate import custom_collate
    from dataset import get_dataset
    from utils.metrics import compute_metrics
    from utils.plot import plot_2d_pde, plot_2d_pde_formal

np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")
logger = getLogger()
metric_to_header = {
    "_l2_error": "rel l2",
    "_mse": "mse",
    "_rmse": "rmse",
    "_l2_error_first_half": "rel l2 1st_half",
    "_l2_error_second_half": "rel l2 2nd_half",
    "_l2_error_step_1": "rel l2 step 1",
    "_l2_error_step_5": "rel l2 step 5",
    "_l2_error_step_10": "rel l2 step 10",
    "_l2_error_int": "rel l2 interior",
}


def get_boundary_mask(patch_size: int, patch_num: int, boundary_width=1):
    """
    Output:
        mask: Tensor (patch_size*patch_num=x_num, x_num)
    """
    single_mask = paddle.ones(patch_size, patch_size)
    single_mask[0:boundary_width] = 0
    single_mask[patch_size - boundary_width :] = 0
    single_mask[:, 0:boundary_width] = 0
    single_mask[:, patch_size - boundary_width :] = 0
    mask = (
        single_mask[None, :, None]
        .expand(patch_num, patch_size, patch_num, patch_size)
        .reshape(patch_num * patch_size, -1)
        .contiguous()
    )
    return mask


class Evaluator(object):
    def __init__(self, trainer, symbol_env):
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.symbol_env = symbol_env
        self.datasets: dict = get_dataset(
            self.params, symbol_env, split="test" if self.params.eval_only else "val"
        )
        self.dataloaders = {
            k: paddle.io.DataLoader(
                dataset=v,
                batch_size=self.params.batch_size_eval,
                num_workers=self.params.num_workers_eval,
                collate_fn=custom_collate(
                    self.params.data.max_output_dimension,
                    symbol_env.pad_index,
                    self.params.data.tie_fields,
                    pad_right=self.params.symbol.pad_right,
                ),
            )
            for k, v in self.datasets.items()
        }
        self.types = self.datasets.keys()
        self.validation_metrics = self.params.validation_metrics_print.split(",")
        self.output_len = self.params.data.t_num - self.params.input_len
        if self.output_len < 10:
            self.validation_metrics.remove("_l2_error_step_10")
        if self.output_len < 5:
            self.validation_metrics.remove("_l2_error_step_5")
        if (
            "_l2_error_int" in self.validation_metrics
            and "patch_num_output" in self.params.model
        ):
            patch_num = self.params.model.patch_num_output
            patch_size = self.params.data.x_num // patch_num
            self.boundary_mask = get_boundary_mask(
                patch_size, patch_num, boundary_width=1
            )[None, None, :, :, None]
            if not self.params.cpu:
                self.boundary_mask = to_cuda(self.boundary_mask)
        else:
            self.validation_metrics.remove("_l2_error_int")
            self.boundary_mask = None

    @paddle.no_grad()
    def evaluate(self):
        params = self.params
        model = self.modules["model"]
        model.eval()
        if params.print_outputs:
            save_folder = os.path.join(params.eval_dump_path, "figures/")
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
        if params.log_eval_plots > 0:
            plot_folder = os.path.join(
                params.eval_dump_path,
                f"epoch_{self.trainer.epoch}_{self.params.local_rank}",
            )
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)
        if params.eval_only or self.trainer.epoch == params.max_epoch - 1:
            input_plot_len = output_plot_len = -1
        else:
            input_plot_len = 1
            output_plot_len = min(2, self.output_len)
        all_results = {}
        more_metrics = []
        for type, loader in self.dataloaders.items():
            num_plotted = 0
            eval_size = 0
            results = defaultdict(list)
            for idx, samples in enumerate(loader):
                bs = len(samples["data"])
                eval_size += bs
                model_input, d = self.trainer.prepare_data(samples, train=False)
                if type in ["incom_ns_arena_u"]:
                    if "output_times" in model_input:
                        model_input["output_times"] = model_input["output_times"][
                            :, : 14 - params.input_len
                        ]
                    if "times" in model_input:
                        model_input["times"] = model_input["times"][:, :14]
                with paddle.amp.autocast(
                    get_amp_device_type(),
                    enabled=bool(params.amp),
                    dtype=paddle.bfloat16,
                ):
                    data_output = model(
                        "generate",
                        model_input["data_input"],
                        model_input["input_times"],
                        model_input["output_times"],
                        model_input.get("symbol_input"),
                        model_input.get("symbol_padding_mask"),
                    )
                if self.params.normalize:
                    if self.params.denormalize_for_loss:
                        data_output = data_output * d["std"] + d["mean"]
                        data_output = data_output * d["data_mask"]
                        data_loss = self.data_loss_fn(
                            data_output, d["data_label"], d["data_mask"]
                        )
                    else:
                        data_output = data_output * d["data_mask"]
                        data_loss = self.data_loss_fn(
                            data_output, d["data_label"], d["data_mask"]
                        )
                        data_output = data_output * d["std"] + d["mean"]
                        d["data_label"] = d["data_label"] * d["std"] + d["mean"]
                else:
                    data_output = data_output * d["data_mask"]
                    data_loss = self.data_loss_fn(
                        data_output, d["data_label"], d["data_mask"]
                    )
                results["data_loss"].extend(data_loss)
                if self.params.data.tie_fields:
                    dim = params.data[type.split(":")[0]].dim
                    data_output = data_output[..., :dim]
                    d["data_label"] = d["data_label"][..., :dim]
                else:
                    valid_mask = d["data_mask"][0, 0, 0, 0].bool()
                    data_output = data_output[..., valid_mask]
                    d["data_label"] = d["data_label"][..., valid_mask]
                cur_result = compute_metrics(
                    data_output,
                    d["data_label"],
                    mask=self.boundary_mask,
                    metrics=self.validation_metrics,
                    batched=True,
                )
                for k in cur_result.keys():
                    results[k].extend(cur_result[k])
                if params.print_outputs:
                    data_output = data_output.float().numpy()
                    if params.data.tie_fields:
                        data_all = samples["data"].numpy()
                    else:
                        data_all = samples["data"][..., valid_mask.cpu()].numpy()
                    t = (
                        samples["t"].numpy()
                        if "t" in samples
                        else self.trainer.t.expand(bs, -1).numpy()
                    )
                    for i in range(bs):
                        index = idx * params.batch_size_eval + i
                        plot_title = "Type {} | Idx {} | {:.4f}".format(
                            type, index, cur_result["_l2_error"][i]
                        )
                        plot_2d_pde_formal(
                            data_output[i],
                            data_all[i],
                            t[i],
                            params.input_len,
                            plot_title,
                            filename=f"{type}_plot_{index}",
                            folder=save_folder,
                            dim=params.data[type.split(":")[0]].dim,
                        )
                if params.log_eval_plots > 0 and num_plotted < params.log_eval_plots:
                    if isinstance(data_output, np.ndarray):
                        output = data_output[0]
                        cur_data = data_all[0]
                        t = t[0]
                    else:
                        output = data_output[0].float().numpy()
                        if params.data.tie_fields:
                            cur_data = samples["data"][0].numpy()
                        else:
                            cur_data = samples["data"][0, ..., valid_mask.cpu()].numpy()
                        t = (
                            samples["t"][0].numpy()
                            if "t" in samples
                            else self.trainer.t[0].numpy()
                        )
                    index = idx * params.batch_size_eval
                    plot_title = "Epoch {} | Type {} | idx {} | {:.4f}".format(
                        self.trainer.epoch, type, index, cur_result["_l2_error"][0]
                    )
                    path = plot_2d_pde(
                        output,
                        cur_data,
                        t,
                        params.input_len,
                        plot_title,
                        filename=f"{type}_plot_{index}",
                        folder=plot_folder,
                        input_plot_len=input_plot_len,
                        output_plot_len=output_plot_len,
                        dim=params.data[type.split(":")[0]].dim,
                    )
                    if params.use_wandb:
                        if self.trainer.epoch <= 4 or (self.trainer.epoch + 1) % 5 == 0:
                            wandb.log(
                                {
                                    "val": {
                                        "epoch": self.trainer.epoch,
                                        f"{type}_plot_{num_plotted}": wandb.Image(path),
                                    }
                                }
                            )
                    num_plotted += 1
                if params.eval_size > 0 and eval_size >= params.eval_size:
                    break
            for k, v in results.items():
                arr = np.array(v)
                results[k] = np.sum(arr)
                if k == "_l2_error":
                    more_metrics.append(
                        [
                            type,
                            len(arr),
                            np.mean(arr),
                            np.std(arr),
                            np.min(arr),
                            np.max(arr),
                            np.median(arr),
                        ]
                    )
            results["size"] = eval_size
            all_results[type] = results
        if params.multi_gpu:
            sorted_keys = None
            for type, results in all_results.items():
                if sorted_keys is None:
                    sorted_keys = sorted(results.keys())
                stats = paddle.Tensor([results[k] for k in sorted_keys])
                stats = sync_tensor(stats)
                results = {k: stats[i].item() for i, k in enumerate(sorted_keys)}
                results["size"] = int(results["size"])
                all_results[type] = results
        total_size = 0
        results_per_type = {}
        stats = defaultdict(list)
        for type, results in all_results.items():
            res_mean_type = {}
            for k, v in results.items():
                if k == "size":
                    res_mean_type[k] = v
                    total_size += v
                else:
                    res_mean_type[k] = v / results["size"]
                    stats[k].append(res_mean_type[k])
            results_per_type[type] = res_mean_type
        stats = {k: np.nanmean(np.array(v)) for k, v in stats.items()}
        headers = ["type", "dim", "size", "data_loss"] + self.validation_metrics
        table = []
        for type, results in results_per_type.items():
            row = [type, self.params.data[type.split(":")[0]].dim]
            for j, k in enumerate(headers[2:]):
                row.append(results[k])
            table.append(row)
        table.append(
            ["AVE_BY_CLASS", "-", "-", stats["data_loss"]]
            + [stats[k] for k in self.validation_metrics]
        )
        headers = list(
            map(lambda s: metric_to_header[s] if s in metric_to_header else s, headers)
        )
        logger.info(
            "Evaluation Stats (total size = {})\n{}".format(
                total_size,
                tabulate(
                    table,
                    headers=headers,
                    tablefmt="grid",
                    floatfmt=[".6f"] * 4 + [".4f"] * len(self.validation_metrics),
                ),
            )
        )
        logger.info(
            "Additional Stats for Rel L2 Error:\n{}".format(
                tabulate(
                    more_metrics,
                    headers=["type", "size", "mean", "std", "min", "max", "median"],
                    tablefmt="grid",
                    floatfmt=".4f",
                )
            )
        )
        return stats, results_per_type

    @paddle.no_grad()
    def rollout(self):
        params = self.params
        model = self.modules["model"]
        model.eval()
        if params.print_outputs:
            save_folder = os.path.join(params.eval_dump_path, "figures/")
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
        if params.log_eval_plots > 0:
            plot_folder = os.path.join(
                params.eval_dump_path,
                f"epoch_{self.trainer.epoch}_{self.params.local_rank}",
            )
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)
        if params.eval_only or self.trainer.epoch == params.max_epoch - 1:
            input_plot_len = output_plot_len = -1
        else:
            input_plot_len = 1
            output_plot_len = min(2, self.output_len)
        all_results = {}
        more_metrics = []
        input_len = self.params.input_len
        for type, loader in self.dataloaders.items():
            num_plotted = 0
            eval_size = 0
            results = defaultdict(list)
            for idx, samples in enumerate(loader):
                bs = len(samples["data"])
                eval_size += bs
                data = samples["data"]
                data_mask = samples["data_mask"]
                t = samples["t"] if self.params.use_raw_time else self.trainer.t
                input_times = t[:, :input_len, None]
                output_times = (
                    t[:, input_len : input_len + 1, None] - input_times[:, -1:]
                )
                data, data_mask, input_times, output_times = to_cuda(
                    data, data_mask, input_times, output_times
                )
                if "symbol_input" in samples:
                    symbol_input = to_cuda(samples["symbol_input"])
                    symbol_mask = to_cuda(samples["symbol_mask"])
                else:
                    symbol_input = symbol_mask = None
                data_all = paddle.zeros_like(data)
                data_all[:, :input_len] = data[:, :input_len]
                data_label = data[:, input_len:]
                for i in range(data.size(1) - input_len):
                    data_input = data_all[:, i : i + input_len]
                    data_input, mean, std = self.prepare_rollout_data(data_input)
                    with paddle.amp.autocast(
                        "cpu" if params.cpu else "cuda",
                        enabled=bool(params.amp),
                        dtype=paddle.bfloat16,
                    ):
                        data_output = model(
                            "generate",
                            data_input=data_input,
                            input_times=input_times,
                            output_times=output_times,
                            symbol_input=symbol_input,
                            symbol_padding_mask=symbol_mask,
                        )
                        data_output = data_output * data_mask
                    if i == 0:
                        first_label = data_label[:, 0:1]
                        first_data_output = data_output
                        if self.params.normalize:
                            if self.params.denormalize_for_loss:
                                first_data_output = first_data_output * std + mean
                                first_data_output = first_data_output * data_mask
                            else:
                                first_label = (first_label - mean) / std
                        data_loss = self.data_loss_fn(
                            first_data_output, first_label, data_mask
                        )
                        results["data_loss"].extend(data_loss)
                    if self.params.normalize:
                        data_output = data_output * std + mean
                    data_all[:, input_len + i] = data_output.squeeze(1)
                data_output = data_all[:, input_len:]
                if self.params.data.tie_fields:
                    dim = params.data[type.split(":")[0]].dim
                    data_output = data_output[..., :dim]
                    data_label = data_label[..., :dim]
                else:
                    valid_mask = data_mask[0, 0, 0, 0].bool()
                    data_output = data_output[..., valid_mask]
                    data_label = data_label[..., valid_mask]
                cur_result = compute_metrics(
                    data_output,
                    data_label,
                    mask=self.boundary_mask,
                    metrics=self.validation_metrics,
                    batched=True,
                )
                for k in cur_result.keys():
                    results[k].extend(cur_result[k])
                if params.print_outputs:
                    data_output = data_output.float().numpy()
                    if params.data.tie_fields:
                        data_all = samples["data"].numpy()
                    else:
                        data_all = samples["data"][..., valid_mask.cpu()].numpy()
                    t = (
                        samples["t"].numpy()
                        if "t" in samples
                        else self.trainer.t.numpy()
                    )
                    for i in range(bs):
                        index = idx * params.batch_size_eval + i
                        plot_title = "Type {} | Idx {} | {:.4f}".format(
                            type, index, cur_result["_l2_error"][i]
                        )
                        plot_2d_pde_formal(
                            data_output[i],
                            data_all[i],
                            t[i],
                            params.input_len,
                            plot_title,
                            filename=f"{type}_plot_{index}",
                            folder=save_folder,
                            dim=params.data[type.split(":")[0]].dim,
                        )
                if params.log_eval_plots > 0 and num_plotted < params.log_eval_plots:
                    if isinstance(data_output, np.ndarray):
                        output = data_output[0]
                        cur_data = data_all[0]
                        t = t[0]
                    else:
                        output = data_output[0].float().numpy()
                        if params.data.tie_fields:
                            cur_data = samples["data"][0].numpy()
                        else:
                            cur_data = samples["data"][0, ..., valid_mask.cpu()].numpy()
                        t = (
                            samples["t"][0].numpy()
                            if "t" in samples
                            else self.trainer.t[0].numpy()
                        )
                    index = idx * params.batch_size_eval
                    plot_title = "Epoch {} | Type {} | idx {} | {:.4f}".format(
                        self.trainer.epoch, type, index, cur_result["_l2_error"][0]
                    )
                    path = plot_2d_pde(
                        output,
                        cur_data,
                        t,
                        params.input_len,
                        plot_title,
                        filename=f"{type}_plot_{index}",
                        folder=plot_folder,
                        input_plot_len=input_plot_len,
                        output_plot_len=output_plot_len,
                        dim=params.data[type.split(":")[0]].dim,
                    )
                    if params.use_wandb:
                        wandb.log(
                            {
                                "val": {
                                    "epoch": self.trainer.epoch,
                                    f"{type}_plot_{num_plotted}": wandb.Image(path),
                                }
                            }
                        )
                    num_plotted += 1
                if params.eval_size > 0 and eval_size >= params.eval_size:
                    break
            for k, v in results.items():
                arr = np.array(v)
                results[k] = np.sum(arr)
                if k == "_l2_error":
                    more_metrics.append(
                        [
                            type,
                            len(arr),
                            np.mean(arr),
                            np.std(arr),
                            np.min(arr),
                            np.max(arr),
                            np.median(arr),
                        ]
                    )
            results["size"] = eval_size
            all_results[type] = results
        if params.multi_gpu:
            sorted_keys = None
            for type, results in all_results.items():
                if sorted_keys is None:
                    sorted_keys = sorted(results.keys())
                stats = paddle.Tensor([results[k] for k in sorted_keys])
                stats = sync_tensor(stats)
                results = {k: stats[i].item() for i, k in enumerate(sorted_keys)}
                results["size"] = int(results["size"])
                all_results[type] = results
        total_size = 0
        results_per_type = {}
        stats = defaultdict(list)
        for type, results in all_results.items():
            res_mean_type = {}
            for k, v in results.items():
                if k == "size":
                    res_mean_type[k] = v
                    total_size += v
                else:
                    res_mean_type[k] = v / results["size"]
                    stats[k].append(res_mean_type[k])
            results_per_type[type] = res_mean_type
        stats = {k: np.nanmean(np.array(v)) for k, v in stats.items()}
        headers = ["type", "dim", "size", "data_loss"] + self.validation_metrics
        table = []
        for type, results in results_per_type.items():
            row = [type, self.params.data[type.split(":")[0]].dim]
            for j, k in enumerate(headers[2:]):
                row.append(results[k])
            table.append(row)
        table.append(
            ["AVE_BY_CLASS", "-", "-", stats["data_loss"]]
            + [stats[k] for k in self.validation_metrics]
        )
        headers = list(
            map(lambda s: metric_to_header[s] if s in metric_to_header else s, headers)
        )
        logger.info(
            "Evaluation Stats (total size = {})\n{}".format(
                total_size,
                tabulate(
                    table,
                    headers=headers,
                    tablefmt="grid",
                    floatfmt=[".6f"] * 4 + [".4f"] * len(self.validation_metrics),
                ),
            )
        )
        logger.info(
            "Additional Stats for Rel L2 Error:\n{}".format(
                tabulate(
                    more_metrics,
                    headers=["type", "size", "mean", "std", "min", "max", "median"],
                    tablefmt="grid",
                    floatfmt=".4f",
                )
            )
        )
        return stats, results_per_type

    def prepare_rollout_data(self, data_input):
        """
        Prepare data for rollout.

        data_input:         Tensor     (bs, input_len, x_num, x_num, dim)
        """
        if self.params.normalize:
            eps = 1e-06
            if self.params.normalize == "meanvar":
                mean = paddle.mean(data_input, axis=(1, 2, 3), keepdim=True)
                std = paddle.std(data_input, axis=(1, 2, 3), keepdim=True) + eps
            elif self.params.normalize == "range":
                max = paddle.amax(data_input, dim=(1, 2, 3), keepdim=True)
                min = paddle.amin(data_input, dim=(1, 2, 3), keepdim=True)
                mean = (max + min) / 2
                std = (max - min) / 2 + eps
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.params.normalize}"
                )
            data_input = (data_input - mean) / std
        else:
            mean = 0
            std = 1
        return data_input, mean, std

    def data_loss_fn(self, data_output, data_label, data_mask):
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
            mask_float = data_mask.cast(loss.dtype)
            nonzero_count = paddle.count_nonzero(
                x=data_mask.expand_as(loss).flatten(1), axis=1
            ).cast(loss.dtype)
            loss = (loss * mask_float).flatten(1).sum(1) / nonzero_count
        else:
            eps = 1e-08
            if self.params.square_loss:
                loss = paddle.nn.functional.mse_loss(
                    input=data_output, label=data_label, reduction="none"
                )
                loss = (loss * data_mask / (weight**2 + eps)).flatten(1).sum(1)
            else:
                loss = paddle.linalg.vector_norm(
                    data_output - data_label, dim=(2, 3), keepdim=True
                )
                loss = (loss / (weight + eps)).flatten(1).sum(1)
        return loss.tolist()
