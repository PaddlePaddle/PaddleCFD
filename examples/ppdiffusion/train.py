import logging
import os
from timeit import default_timer

import hydra
import numpy as np
import paddle
from metrics import EnsembleMetrics
from models import LitEma
from omegaconf import DictConfig
from process import DYffusion, Interpolation
from utils import (
    AverageMeterDict,
    get_dataloader,
    get_optimizer,
    get_scheduler,
    initialize_models,
    save_arrays_as_line_plot,
    set_seed,
)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.process = cfg.process.lower()
        assert self.process in ["interpolation", "dyffusion"]

        # check ckpt of interpolation
        interp_ckpt = getattr(self.cfg.INTERPOLATION, "ckpt_no_suffix", None)
        assert interp_ckpt is not None, "Error: interpolation ckpt should not be None in dyffusion process."
        self.ckpt = interp_ckpt
        self.resume_ep = 0
        # set accumulate_steps
        self.accumulate_steps = getattr(cfg.TRAIN, "accumulate_steps", 1)
        self.calc_ensemble_metrics = True
        self.calc_batch_metrics = False
        self.visual_metrics = False

        self.init_model()
        self.init_dataloader()
        self.init_opt_sched()
        self.init_loss_fn()
        self.init_metric_fn()
        self.init_ema()
        self.init_meters()

        if self.process == "interpolation":
            self.concat_fn = self.process_obj.concat_results
        elif self.process == "dyffusion":
            self.concat_fn = self.process_obj.interp_obj.concat_results

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

    def init_dataloader(self):
        # initialize dataloader
        dataloaders = get_dataloader(self.cfg.DATA, ["train", "val", "test"])
        self.dataloader_train = dataloaders["train"]
        self.dataloader_val = dataloaders["val"]
        self.dataloader_test = dataloaders["test"]

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

    def init_metric_fn(self):
        self.metric_fn = EnsembleMetrics(mean_over_samples=self.calc_ensemble_metrics)

    def init_ema(self):
        self.ema = LitEma(self.model_to_save, decay=self.cfg.EVAL.ema.decay)

    def init_meters(self):
        self.train_meter = AverageMeterDict()
        self.val_meter = AverageMeterDict()
        self.val_meter_verbose = AverageMeterDict()

    def eval_and_save(self, save_path="./test"):
        with self.model_to_save.dropout_controller(enable=self.cfg.EVAL.enable_infer_dropout):
            self.validate(**{"dataloader": self.dataloader_val})
        paddle.save(self.model_to_save.state_dict(), f"{save_path}.pdparams")
        paddle.save(self.optimizer.state_dict(), f"{save_path}.pdopt")

    def train(self):
        self.model_to_save.train()
        for ep in range(self.resume_ep, self.cfg.TRAIN.epochs):
            t1 = default_timer()
            for i, data_dict in enumerate(self.dataloader_train):
                preds, targets = self.process_obj.forward(data_dict)
                loss = self.process_obj.get_loss(preds, targets, self.loss_fn)
                loss.backward()
                if (i + 1) % self.accumulate_steps == 0:
                    self.optimizer.step()
                    self.optimizer.clear_grad(set_to_zero=False)
                self.train_meter.update({"l1": loss})

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
            if (ep + 1) % self.cfg.TRAIN.save_freq == 0 or ep == self.cfg.TRAIN.epochs - 1 or (ep + 1) == 1:
                self.eval_and_save(save_path=f"{self.cfg.output_dir}/{self.process}_{ep}")

    def get_batch_metrics(self, return_dict):
        preds_batch = paddle.stack([return_dict[k] for k in return_dict if "preds" in k], axis=-5)
        targets_batch = paddle.stack([return_dict[k] for k in return_dict if "targets" in k], axis=-5)
        metric_dict = self.metric_fn.metric(preds_batch, targets_batch, crps_member_dim=1)
        for name, metric in metric_dict.items():
            self.val_meter.update({f"{name}_mean(one_batch)": paddle.mean(metric)})
        return metric_dict

    def get_ensemble_metrics(self, return_lst):
        results_concat = self.concat_fn(return_lst)
        # Go through all predictions and compute metrics (i.e. over preds for each time step)
        t_steps = set([key.split("_")[0] for key in results_concat.keys()])
        for t_step in t_steps:
            pred = results_concat[f"{t_step}_preds"]
            target = results_concat[f"{t_step}_targets"]
            metric_dict = self.metric_fn.metric(pred, target)
            for name, metric in metric_dict.items():
                self.val_meter.update({f"{name}_mean(all_ts)": metric})
                self.val_meter_verbose.update({f"{t_step}_{name}": metric})

    def visualize(self, metric_dict=None):
        if metric_dict is not None:
            timesteps = range(1, self.cfg.EVAL.prediction_horizon + 1)
            save_arrays_as_line_plot(
                np.array(timesteps),
                metric_dict,
                save_dir=self.cfg.EVAL.save_dir,
                x_label="time",
                y_label="metrics",
            )

    def validate(self, **kwargs):
        metric_dict = None
        dataloader = kwargs.pop("dataloader")
        t1 = default_timer()
        with self.ema.ema_scope(use_ema=self.cfg.EVAL.ema.use_ema, context="Validation"):
            return_lst = []
            for i, data_dict in enumerate(dataloader):
                return_dict = self.process_obj.eval(data_dict, **kwargs)
                return_lst.append(return_dict)
                if self.calc_batch_metrics:
                    metric_dict = self.get_batch_metrics(return_dict)
                if self.visual_metrics:
                    self.visualize(metric_dict)

            if self.calc_ensemble_metrics:
                self.get_ensemble_metrics(return_lst)
        t2 = default_timer()

        logging.info(
            "[Eval] time: %.2fs [Metric] %s",
            t2 - t1,
            self.val_meter.message(),
        )
        if self.cfg.EVAL.verbose:
            logging.info(self.val_meter_verbose.message_verbose())

    def test(self):
        # update cfg
        self.cfg.EVAL.batch_size = self.cfg.DATA.dataloader.batch_size.test
        self.cfg.DATA.dataset.horizon = self.cfg.EVAL.prediction_horizon
        self.cfg.INTERPOLATION.num_predictions = self.cfg.EVAL.prediction_num_predictions
        self.cfg.FORECASTING.num_predictions = self.cfg.EVAL.prediction_num_predictions
        self.cfg.EVAL.verbose = False
        self.model_to_save.eval()

        kwargs = {"dataloader": self.dataloader_test}
        if self.process == "dyffusion":
            kwargs.update({"enable_ar": enable_ar, "ar_steps": self.cfg.EVAL.autoregressive_steps})
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

    trainer = Trainer(cfg)
    if cfg.mode == "train":
        logging.info(f"######## Training {cfg.process}... ########")
        trainer.train()
    elif cfg.mode == "test":
        logging.info(f"######## Testing {cfg.process}... ########")
        trainer.test()
    else:
        raise ValueError(f"cfg.mode should in ['train', 'test'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
