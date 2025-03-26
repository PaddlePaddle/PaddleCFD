import logging
import os
from timeit import default_timer

import hydra
import numpy as np
import paddle
from metrics import EnsembleMetrics
from models import LitEma, SimpleUnet
from omegaconf import DictConfig
from process import Interpolation
from utils import AverageMeterDict, get_dataloader, get_optimizer, get_scheduler, set_seed


def train(cfg: DictConfig, process):
    # initialize process class
    if process.lower() == "interpolation":
        process_obj = Interpolation(cfg)
        model_cfg = process_obj.model_cfg_transform(cfg.MODEL)

    # initialize dataloader
    dataloaders = get_dataloader(cfg.DATA, ["train", "val"])
    dataloader_train = dataloaders["train"]
    dataloader_val = dataloaders["val"]

    # initialize model
    model = SimpleUnet(**model_cfg)
    # load ckpt
    ckpt = getattr(cfg.TRAIN, "ckpt_no_suffix", None)
    if ckpt:
        state_dict = paddle.load(f"{ckpt}.pdparams")
        model.set_state_dict(state_dict)
    model.train()
    process_obj.set_model(model)

    # initialize optimizer and scheduler
    optimizer = get_optimizer(cfg.TRAIN.optim, model.parameters(), ckpt)
    # set scheduler
    cfg_sched = getattr(cfg.TRAIN, "sched", None)
    if cfg_sched:
        scheduler = get_scheduler(cfg_sched)
        optimizer.set_lr_scheduler(scheduler)
        resume_ep = scheduler.last_epoch
    else:
        resume_ep = 0
    assert cfg.TRAIN.epochs > resume_ep, f"Error: training epochs {cfg.TRAIN.epochs} < resume epoch {resume_ep} now."
    logging.info(f"lr of epoch {resume_ep} is {optimizer.get_lr()}")

    # initialize loss function
    loss_fn = paddle.nn.MSELoss("mean")
    # set accumulate_steps
    accumulate_steps = getattr(cfg.TRAIN, "accumulate_steps", 1)

    for ep in range(cfg.TRAIN.epochs):
        t1 = default_timer()
        train_meter = AverageMeterDict()
        for i, data_dict in enumerate(dataloader_train):
            preds, targets = process_obj.forward(data_dict)
            loss = process_obj.get_loss(preds, targets, loss_fn)
            loss.backward()
            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.clear_grad(set_to_zero=False)
            train_meter.update({"mse": loss})

        if cfg_sched:
            scheduler.step()
        t2 = default_timer()

        logging.info(
            "[Train][Epoch %d/%d] time: %.2fs, lr: %g [Loss] %s",
            ep + 1,
            cfg.TRAIN.epochs,
            t2 - t1,
            optimizer.get_lr(),
            train_meter.message(),
        )

        # eval and save the weights
        if (ep + 1) % cfg.TRAIN.save_freq == 0 or ep == cfg.TRAIN.epochs - 1 or (ep + 1) == 1:
            with process_obj.model.dropout_controller(enable=cfg.EVAL.enable_infer_dropout):
                validate(cfg.EVAL, dataloader_val, process_obj)
            paddle.save(process_obj.model.state_dict(), f"{cfg.output_dir}/{process}_{ep}.pdparams")
            paddle.save(optimizer.state_dict(), f"{cfg.output_dir}/{process}_{ep}.pdopt")


def validate(cfg_eval, dataloader, process_obj, calc_ensemble_metrics: bool = True):
    # initialize metric function
    # metric_fn = paddle.nn.MSELoss("mean")
    metric_ensemble = EnsembleMetrics()

    # set EMA
    ema = LitEma(process_obj.model, decay=cfg_eval.ema.decay)

    t1 = default_timer()
    val_meter = AverageMeterDict()
    val_meter_verbose = AverageMeterDict()

    with ema.ema_scope(use_ema=cfg_eval.ema.use_ema, context="Validation"):
        return_lst = []
        for i, data_dict in enumerate(dataloader):
            return_dict = process_obj.eval(data_dict)
            return_lst.append(return_dict)

        if calc_ensemble_metrics:
            results_concat = process_obj.concat_results(return_lst)
            # Go through all predictions and compute metrics (i.e. over preds for each time step)
            for t_step in np.arange(1, process_obj.horizon):
                pred = results_concat[f"t{t_step}_preds"]
                target = results_concat[f"t{t_step}_targets"]
                metric_dict = metric_ensemble.metric(pred, target)
                for name, metric in metric_dict.items():
                    val_meter.update({f"{name}_mean(all_ts)": metric})
                    val_meter_verbose.update({f"t{t_step}_{name}": metric})

    t2 = default_timer()

    logging.info(
        "[Eval] time: %.2fs [Metric] %s",
        t2 - t1,
        val_meter.message(),
    )
    if cfg_eval.verbose:
        logging.info(val_meter_verbose.message_verbose())


def test(cfg: DictConfig, process):
    # update cfg
    cfg.EVAL.batch_size = cfg.DATA.dataloader.batch_size.test
    cfg.DATA.dataset.horizon = cfg.EVAL.prediction_horizon
    cfg.INTERPOLATION.num_predictions = cfg.EVAL.prediction_num_predictions
    cfg.EVAL.verbose = True

    # initialize process class
    if process.lower() == "interpolation":
        process_obj = Interpolation(cfg)
        model_cfg = process_obj.model_cfg_transform(cfg.MODEL)

    # initialize dataloader
    dataloaders = get_dataloader(cfg.DATA, ["test"])
    dataloader_test = dataloaders["test"]

    # initialize model
    model = SimpleUnet(**model_cfg)
    # load ckpt
    ckpt = getattr(cfg.EVAL, "ckpt_no_suffix", None)
    assert ckpt, "Error: ckpt is None."
    state_dict = paddle.load(f"{ckpt}.pdparams")
    model.set_state_dict(state_dict)
    process_obj.set_model(model)

    with process_obj.model.dropout_controller(enable=cfg.EVAL.enable_infer_dropout):
        validate(cfg.EVAL, dataloader_test, process_obj)


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
        logging.info(f"######## Training {cfg.process} with model {cfg.MODEL.model_name} ... ########")
        train(cfg, cfg.process)
    elif cfg.mode == "test":
        logging.info(f"######## Testing {cfg.process} with model {cfg.MODEL.model_name} ... ########")
        test(cfg, cfg.process)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'test'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
