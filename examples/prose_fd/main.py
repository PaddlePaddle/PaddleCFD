import os

# CINN 动转静开关
if os.environ.get("PROSE_TO_STATIC", "0") == "1":
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"

from pathlib import Path

import hydra
import numpy as np
import paddle
import wandb
from omegaconf import DictConfig, OmegaConf

from ppcfd.models.prose_fd import build_model, SymbolicEnvironment
from ppcfd.models.prose_fd import runtime as prose_runtime

try:
    from examples.prose_fd.evaluate import Evaluator, metric_to_header
    from examples.prose_fd.trainer import Trainer
    from examples.prose_fd.utils.misc import initialize_exp, set_seed
    from examples.prose_fd.utils.mode import init_distributed_mode
except ImportError:
    from evaluate import Evaluator, metric_to_header
    from trainer import Trainer
    from utils.misc import initialize_exp, set_seed
    from utils.mode import init_distributed_mode

np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(params: DictConfig):
    if params.dryrun:
        print("Debugging run...")
        params.max_epoch = 1
        params.n_steps_per_epoch = 5
        params.debug = True
        params.exp_name = "debug"
        params.use_wandb = 0
        params.wandb.entity = None
        params.save_periodic = 0
        params.log_periodic = 1
        if not params.batch_size_eval:
            params.batch_size_eval = int(1.5 * params.batch_size)
        params.eval_size = params.batch_size_eval * 2
        params.base_seed = 1
        params.log_eval_plots = -1
    if params.eval_only:
        assert params.eval_from_exp is not None and params.eval_from_exp != ""
        if os.path.exists(params.eval_from_exp + "/checkpoint.pth"):
            params.reload_model = params.eval_from_exp + "/checkpoint.pth"
        else:
            assert os.path.exists(params.eval_from_exp)
            params.reload_model = params.eval_from_exp
        if params.overfit_test and params.exp_id:
            params.exp_id = params.exp_id + "_train"
        if params.eval_single_file and params.exp_id:
            params.exp_id = params.exp_id + "_file"
    init_distributed_mode(params)
    prose_runtime.set_runtime_device(params.runtime_device)
    if params.optim.type in ["adamw", "adan"]:
        if params.optim.warmup is not None and params.optim.warmup < 1:
            params.optim.warmup = max(
                1,
                int(
                    params.optim.warmup
                    * params.max_epoch
                    * params.n_steps_per_epoch
                    // params.accumulate_gradients
                ),
            )
        params.optim.max_iters = (
            params.max_epoch * params.n_steps_per_epoch // params.accumulate_gradients
        )
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    logger = initialize_exp(params)
    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)
    if not params.is_master:
        params.use_wandb = False
    if params.use_wandb:
        if not params.wandb.id:
            params.wandb.id = wandb.util.generate_id()
        wandb.init(
            project=params.wandb.project if params.wandb.project else params.exp_name,
            resume="allow",
            id=params.wandb.id,
            name=params.wandb.name,
            entity=params.wandb.entity,
            notes=params.wandb.notes,
            dir=params.dump_path,
        )
        config_d = OmegaConf.to_container(params, resolve=True, throw_on_missing=True)
        config = {"params": {}}
        keys_to_separate = ["model", "data", "optim", "wandb", "symbol"]
        for k, v in config_d.items():
            if k in keys_to_separate:
                config[k] = v
            else:
                config["params"][k] = v
        wandb.config.update(config, allow_val_change=True)
    if params.eval_only:
        set_seed(params.test_seed)
    else:
        set_seed(params.base_seed)
    symbol_env = SymbolicEnvironment(params.symbol)
    modules = build_model(params, params.model, params.data, symbol_env)
    trainer = Trainer(modules, params, symbol_env)
    evaluator = Evaluator(trainer, symbol_env)
    if params.eval_only:
        if params.rollout:
            stats, _ = evaluator.rollout()
        else:
            stats, _ = evaluator.evaluate()
        s = "Eval | data loss = {:.6f}".format(stats["data_loss"])
        for metric in evaluator.validation_metrics:
            s += " | {} = {:.6f}".format(metric_to_header[metric], stats[metric])
        logger.info(s)
        max_mem = prose_runtime.max_memory_allocated_mb()
        if max_mem is not None:
            logger.info(" MEM: {:.2f} MB ".format(max_mem))
        exit()
    while trainer.epoch < params.max_epoch:
        logger.info(f"============ Starting epoch {trainer.epoch} ... ============")
        trainer.inner_epoch = 0
        while trainer.inner_epoch < trainer.n_steps_per_epoch:
            trainer.iter()
            if (
                params.log_periodic > 0
                and trainer.inner_epoch % params.log_periodic == 0
            ):
                data_loss = trainer.data_loss / params.log_periodic
                logger.info(
                    "Epoch {} | step {} | data loss = {:.8f}".format(
                        trainer.epoch, trainer.inner_epoch, data_loss
                    )
                )
                if params.use_wandb:
                    wandb.log(
                        {
                            "train": {
                                "data_loss": data_loss,
                                "epoch": trainer.epoch,
                                "step": trainer.n_total_iter,
                            }
                        }
                    )
                trainer.data_loss = 0.0
        logger.info(f"============ End of epoch {trainer.epoch} ============")
        trainer.save_periodic()
        logger.info(
            "====== STARTING EVALUATION (multi-gpu: {}) =======".format(
                params.multi_gpu
            )
        )
        stats, results_per_type = evaluator.evaluate()
        s = "Epoch {} Eval | data loss = {:.6f}".format(
            trainer.epoch, stats["data_loss"]
        )
        for metric in evaluator.validation_metrics:
            s += " | {} = {:.6f}".format(metric_to_header[metric], stats[metric])
        logger.info(s)
        if params.use_wandb:
            stats["epoch"] = trainer.epoch
            wandb_log = {"val": {k.strip("_"): v for k, v in stats.items()}}
            if params.wandb.log_per_type:
                for type, results in results_per_type.items():
                    wandb_log["val"][type] = {
                        k.strip("_"): v
                        for k, v in results.items()
                        if k in ["_l2_error", "data_loss"]
                    }
            wandb.log(wandb_log)
        trainer.save_best_model(stats)
        trainer.end_epoch()
    max_mem = prose_runtime.max_memory_allocated_mb()
    if max_mem is not None:
        logger.info(" MEM: {:.2f} MB ".format(max_mem))
    if params.multi_gpu:
        paddle.distributed.destroy_process_group()
    if params.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
