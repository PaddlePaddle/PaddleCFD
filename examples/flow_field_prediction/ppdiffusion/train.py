import logging
import os
import warnings
from timeit import default_timer

import hydra
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from omegaconf import DictConfig
from utils import AverageMeterDict
from utils import get_dataloader
from utils import get_optimizer
from utils import get_scheduler
from utils import initialize_models
from utils import save_arrays_as_gif
from utils import save_arrays_as_line_plot
from utils import set_seed

from ppcfd.models.ppdiffusion import auto_adapt_dataparallel
from ppcfd.models.ppdiffusion.metrics import EnsembleMetrics
from ppcfd.models.ppdiffusion.modules import LitEma
from ppcfd.models.ppdiffusion.process import DYffusion
from ppcfd.models.ppdiffusion.process import Interpolation


warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.process = cfg.process.lower()
        assert self.process in ["interpolation", "dyffusion"]

        self.resume_ep = 0
        self.accumulate_steps = getattr(cfg.TRAIN, "accumulate_steps", 1)
        self.metric_fns = getattr(cfg.TRAIN, "metric_fns", ["mse"])
        self.calc_batch_metrics = getattr(cfg.EVAL, "calc_batch_metrics", False)
        calc_ensemble_metrics = getattr(cfg.EVAL, "calc_ensemble_metrics", True)
        self.calc_ensemble_metrics = False if self.calc_batch_metrics else calc_ensemble_metrics
        self.visual_metrics = getattr(cfg.EVAL, "visual_metrics", False)

        self.init_model()
        if self.process == "interpolation":
            self.concat_fn = self.process_obj.concat_results
            self.ckpt = getattr(cfg.INTERPOLATION, "checkpoint", None)
        elif self.process == "dyffusion":
            self.concat_fn = self.process_obj.interp_obj.concat_results
            self.ckpt = getattr(cfg.FORECASTING, "checkpoint", None)

        self.init_opt_sched()
        self.init_loss_fn()
        self.init_metric_fn()
        self.init_ema()
        self.init_meters()

        self.world_size = dist.get_world_size()
        if self.world_size > 1:
            fleet.init(is_collective=True)
            self.model_to_save = fleet.distributed_model(self.model_to_save)
            self.model_to_save = auto_adapt_dataparallel(self.model_to_save, ["dropout_controller"])
            self.optimizer = fleet.distributed_optimizer(self.optimizer)

    def init_model(self):
        if self.process == "interpolation":
            self.process_obj = Interpolation(self.cfg)
            model_interp = initialize_models(self.cfg, models_lst=["interp"], interp_obj=self.process_obj)[0]
            self.process_obj.set_model(model_interp)
            self.model_to_save = model_interp
        elif self.process == "dyffusion":
            self.process_obj = DYffusion(self.cfg)
            # initialize models
            model_interp, model_forecast = initialize_models(
                self.cfg,
                models_lst=["interp", "forecast"],
                interp_obj=self.process_obj.interp_obj,
                forecast_obj=self.process_obj.forecast_obj,
            )
            model_interp.eval()
            # update models
            self.process_obj.init_models(model_interp, model_forecast)
            self.model_to_save = model_forecast

    def init_opt_sched(self):
        # initialize optimizer and scheduler
        self.optimizer = get_optimizer(self.cfg.TRAIN.optim, self.model_to_save.parameters(), self.ckpt)
        # set scheduler
        cfg_sched = getattr(self.cfg.TRAIN, "sched", None)
        if cfg_sched:
            self.scheduler = get_scheduler(cfg_sched)
            optimizer.set_lr_scheduler(scheduler)
            self.resume_ep = scheduler.last_epoch
        else:
            self.scheduler = None
        assert (
            self.cfg.TRAIN.epochs > self.resume_ep
        ), f"Error: training epochs {self.cfg.TRAIN.epochs} < resume epoch {self.resume_ep} now."
        logging.info(f"lr of epoch {self.resume_ep} is {self.optimizer.get_lr()}")

    def init_loss_fn(self):
        if self.process == "interpolation":
            loss_name = self.cfg.INTERPOLATION.loss_fn
        elif self.process == "dyffusion":
            loss_name = self.cfg.FORECASTING.loss_fn

        if loss_name == "l1":
            self.loss_fn = paddle.nn.L1Loss("mean")
        elif loss_name == "mse":
            self.loss_fn = paddle.nn.MSELoss("mean")
        else:
            logging.info(f"{loss_name} is not supported now. Auto switch to MSE loss")
            self.loss_fn = paddle.nn.MSELoss("mean")
        self.loss_name = loss_name

    def init_metric_fn(self):
        self.metric_fn = EnsembleMetrics(mean_over_samples=self.calc_ensemble_metrics, metric_fns=self.metric_fns)

    def init_ema(self):
        self.ema = LitEma(self.model_to_save, decay=self.cfg.EVAL.ema.decay)

    def init_meters(self):
        self.train_meter = AverageMeterDict()
        self.val_meter = AverageMeterDict()
        self.val_meter_verbose = AverageMeterDict()

    def eval_and_save(self, dataloader, save_path="./test"):
        self.model_to_save.eval()
        with self.model_to_save.dropout_controller(enable=self.cfg.EVAL.enable_infer_dropout):
            self.validate(**{"dataloader": dataloader})
        paddle.save(self.model_to_save.state_dict(), f"{save_path}.pdparams")
        paddle.save(self.optimizer.state_dict(), f"{save_path}.pdopt")
        self.model_to_save.train()

    def train(self):
        # initialize dataloader
        dataloaders = get_dataloader(self.cfg.DATA, ["train", "val"])
        dataloader_train = dataloaders["train"]
        dataloader_val = dataloaders["val"]

        if self.cfg.TRAIN.enable_amp:
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        self.model_to_save.train()
        for ep in range(self.resume_ep, self.cfg.TRAIN.epochs):
            t1 = default_timer()
            self.train_meter.reset()
            for i, data_dict in enumerate(dataloader_train):

                with paddle.amp.auto_cast(enable=self.cfg.TRAIN.enable_amp, level="O1"):
                    preds, targets = self.process_obj.forward(data_dict)

                loss = self.process_obj.get_loss(preds, targets, self.loss_fn)
                (loss if not self.cfg.TRAIN.enable_amp else scaler.scale(loss)).backward()

                if (i + 1) % self.accumulate_steps == 0 or (i + 1) == len(dataloader_train):
                    if self.cfg.TRAIN.enable_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.clear_grad(set_to_zero=False)

                self.train_meter.update({self.loss_name: loss})

            if self.scheduler is not None:
                self.scheduler.step()
            t2 = default_timer()

            logging.info(
                "[Train][Epoch %d/%d] time: %.2fs, lr: %g [Loss] %s",
                ep + 1,
                self.cfg.TRAIN.epochs,
                t2 - t1,
                self.optimizer.get_lr(),
                self.train_meter.message(),
            )

            # eval and save the weights
            if (ep + 1) % self.cfg.TRAIN.save_freq == 0 or ep == self.cfg.TRAIN.epochs - 1:  # or (ep + 1) == 1:
                self.eval_and_save(dataloader=dataloader_val, save_path=f"{self.cfg.output_dir}/{self.process}_{ep}")

    def get_batch_metrics(self, return_dict):
        preds_batch = paddle.stack([return_dict[k] for k in return_dict if "preds" in k], axis=-5)
        targets_batch = paddle.stack([return_dict[k] for k in return_dict if "targets" in k], axis=-5)
        metric_dict = self.metric_fn.metric(preds_batch, targets_batch, crps_member_dim=1)
        for name, metric in metric_dict.items():
            self.val_meter.update({f"{name}_mean(one_batch)": paddle.mean(metric)})
        return metric_dict, preds_batch, targets_batch

    def get_ensemble_metrics(self, return_lst):
        results_concat = self.concat_fn(return_lst)
        # Go through all predictions and compute metrics (i.e. over preds for each time step)
        t_steps = sorted({int(key.split("_")[0][1:]) for key in results_concat.keys()})
        for t_step in t_steps:
            pred = results_concat[f"t{t_step}_preds"]
            target = results_concat[f"t{t_step}_targets"]
            metric_dict = self.metric_fn.metric(pred, target)
            for name, metric in metric_dict.items():
                self.val_meter.update({f"{name}_mean(all_ts)": metric})
                self.val_meter_verbose.update({f"t{t_step}_{name}": metric})

    def visualize(self, metric_dict=None, preds=None, targets=None, extra_info=""):
        save_dir = os.path.join(self.cfg.EVAL.save_dir, "visual")
        if metric_dict is not None:
            timesteps = next(iter(metric_dict.values())).shape[0] + 1
            timesteps = range(1, timesteps)
            save_arrays_as_line_plot(
                np.array(timesteps),
                metric_dict,
                save_dir=save_dir,
                x_label="time",
                y_label="metrics",
                extra_info=extra_info,
            )
        if preds is not None and targets is not None:
            save_arrays_as_gif(
                preds.numpy(),
                targets.numpy(),
                save_dir=save_dir,
                titles=["u", "v", "p"],
                extra_info=extra_info,
            )

    def validate(self, **kwargs):
        metric_dict, preds_batch, targets_batch = None, None, None
        self.val_meter.reset()
        self.val_meter_verbose.reset()
        dataloader = kwargs.pop("dataloader")

        logging.info("[Starting validating...]")
        t1 = default_timer()
        with self.ema.ema_scope(use_ema=self.cfg.EVAL.ema.use_ema, context="Validation"):
            return_lst = []
            for i, data_dict in enumerate(dataloader):
                return_dict = self.process_obj.eval(data_dict, **kwargs)
                return_lst.append(return_dict)
                if self.calc_batch_metrics:
                    metric_dict, preds_batch, targets_batch = self.get_batch_metrics(return_dict)
                if i == 0:
                    logging.info(
                        "Only save the visualization of the 0th sample. "
                        "The validation process is still in progress after printing. "
                        "Please wait patiently."
                    )
                    if self.visual_metrics:
                        preds_mean = paddle.mean(preds_batch, axis=0)
                        self.visualize(metric_dict, preds_mean, targets_batch, extra_info=f"_{i}")

            if self.calc_ensemble_metrics:
                self.get_ensemble_metrics(return_lst)
        t2 = default_timer()

        logging.info(
            "[Eval] time: %.2fs [Metric] %s",
            t2 - t1,
            self.val_meter.message(),
        )
        if self.cfg.EVAL.verbose and self.calc_ensemble_metrics:
            logging.info(self.val_meter_verbose.message_verbose())

    def test(self):
        # initialize dataloader
        dataloaders = get_dataloader(self.cfg.DATA, ["test"])
        dataloader_test = dataloaders["test"]

        # update metric_fns
        self.metric_fns = getattr(self.cfg.EVAL, "metric_fns", ["mse"])
        self.init_metric_fn()

        kwargs = {"dataloader": dataloader_test}
        if self.process == "dyffusion":
            kwargs.update(
                {
                    "pred_horizon": self.cfg.EVAL.prediction_horizon,
                    "enable_ar": self.cfg.EVAL.enable_ar,
                    "ar_steps": self.cfg.EVAL.autoregressive_steps,
                }
            )
        self.model_to_save.eval()
        with self.model_to_save.dropout_controller(enable=self.cfg.EVAL.enable_infer_dropout):
            self.validate(**kwargs)


@hydra.main(config_path="configs/", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    # logging setting
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"{cfg.mode}.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if cfg.mode == "train":
        logging.info(f"######## Training {cfg.process}... ########")
        trainer = Trainer(cfg)
        trainer.train()
    elif cfg.mode == "test":
        logging.info(f"######## Testing {cfg.process}... ########")
        # update cfg
        cfg.EVAL.batch_size = cfg.DATA.dataloader.batch_size.test
        cfg.DATA.dataset.horizon = cfg.EVAL.prediction_horizon
        cfg.INTERPOLATION.horizon = cfg.EVAL.prediction_horizon
        cfg.FORECASTING.horizon = cfg.EVAL.prediction_horizon
        cfg.INTERPOLATION.num_predictions = cfg.EVAL.prediction_num_predictions
        cfg.FORECASTING.num_predictions = cfg.EVAL.prediction_num_predictions
        cfg.SAMPLING.num_timesteps = cfg.EVAL.prediction_horizon
        cfg.EVAL.verbose = True

        trainer = Trainer(cfg)
        trainer.test()
    else:
        raise ValueError(f"cfg.mode should in ['train', 'test'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
