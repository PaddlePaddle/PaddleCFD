import glob
import logging
import os
from timeit import default_timer
from typing import List
import pickle
import random

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from paddle.distributed import fleet
from tqdm import tqdm

#from dataset_creation_nurbs import *
from dataset_creation_sim import preprocess_data_airfrans, FoilDataset, complete_field_prediction
from average_meter import AverageMeterDict
from ppcfd.models.ppkan import kan
from mlp import DeepONet
from visualization import plot_metrics, plot_predictions, parse_log_file

def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)

def train(cfg: DictConfig, with_val=False):
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"{cfg.mode}.log"),
        level=logging.INFO,
        format="%(ascttime)s:%(levelname)s:%(message)s",
    )
    # initialize model
    if cfg.model == "DeepONet":
        model = DeepONet(**cfg.MLPMODEL)
    elif cfg.model == "KANONet":
        model = kan.KANONet(**cfg.KANMODEL)
    if cfg.checkpoint:
        param_dict = paddle.load(f"{cfg.checkpoint}.pdparams")
        model.set_state_dict(param_dict)
    model.train()
    # initialize optimizer and scheduler
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=cfg.lr, weight_decay=1e-4
    )
    if cfg.enable_ddp:
        model = fleet.distributed(model)
        optimizer = fleet.distributed(optimizer)
        # load optimizer
    resume_ep = cfg.resume_ep
    if cfg.checkpoint and os.path.exists(f"{cfg.checkpoint}.pdopt"):
        optim_dict = paddle.load(f"{cfg.checkpoint}.pdopt")
        optimizer.set_state_dict(optim_dict)
        resume_ep = optim_dict["LR_Scheduler"]["last_epoch"]
    error_msg = (
        f"training epochs {cfg.epochs} should be greater than resume epoch, "
        f"which is {resume_ep} now."
    )
    assert cfg.epochs > resume_ep, error_msg

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=optimizer.get_lr(),
        T_max=cfg.epochs,
        last_epoch=resume_ep,
    )
    optimizer.set_lr_scheduler(scheduler)
    logging.info(f"lr of {resume_ep+1} is {optimizer.get_lr()}")

    # set loss_fn
    loss_fn = paddle.nn.MSELoss()

    # Load and process data
    t1 = default_timer()
    train_set, train_loader, invel_mean, invel_std, lables_mean, labels_std, _, _ = preprocess_data_airfrans(**cfg.DATA, train=True)
    if with_val:
        test_set, test_loader, test_invel_mean, test_invel_std, test_lables_mean, test_labels_std, X_test, Y_test = preprocess_data_airfrans(**cfg.DATA, train=False)
        
    t2 = default_timer()
    logging.info(f"Loading data took {t2 - t1:.2f} seconds.")

    # training
    logging.info(f"Start training {cfg.model}...")
    best_metric = 10.0
    for ep in range(resume_ep + 1, cfg.epochs):
        t1 = default_timer()
        train_meter = AverageMeterDict()
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.clear_grad(set_to_zero=False)
            branch1 = x_batch[:, :cfg.MLPMODEL.branch1_in].astype(cfg.dtype) # Airfoil coord[0:400】, Normalized inlet velocity[400, 401], sdf [402]
            trunk = x_batch[:, cfg.MLPMODEL.branch1_in:].astype(cfg.dtype) # Trunk coord [403, 404]
            Y_train = y_batch.astype(cfg.dtype) # Normalized Lables: Pressure/rho, velocity_x, velocity_y, viscosity_turbulence
            Y_pred = model({
                'branch1': branch1, 
                'trunk': trunk,
            })
            loss = loss_fn(Y_pred, Y_train)
            loss.backward()
            optimizer.step()
            train_meter.update({"loss": loss})

        scheduler.step()
        t2 = default_timer()
        msg = ""
        for k, v in train_meter.avg.items():
            msg += f"{v.item():.4f}({k}), "
        logging.info(
            "[Train][Epoch %d/%d] time: %.2fs, lr: %g [Loss] %s",
            ep + 1,
            cfg.epochs,
            t2 - t1,
            optimizer.get_lr(),
            msg
        )

        if with_val and (ep+1) % cfg.save_freq == 0 or (ep+1) == 1:
            metric = valid(cfg, test_loader, model)
            if metric <= best_metric:
                best_metric = metric
                paddle.save(
                    model.state_dict(), f"{cfg.output_dir}/{cfg.model}_best.pdparams"
                )
                if optimizer:
                    paddle.save(
                        optimizer.state_dict(), f"{cfg.output_dir}/{cfg.model}_best.pdopt"
                    )
        
        if (ep + 1) % cfg.save_freq == 0 or ep == cfg.epochs - 1:
            paddle.save(
                model.state_dict(), f"{cfg.output_dir}/{cfg.model}_latest.pdparams"
            )
            if optimizer:
                paddle.save(
                    optimizer.state_dict(), f"{cfg.output_dir}/{cfg.model}_latest.pdopt"
                )
    
    # visulization metric curve and prediction
    train_epochs, train_losses, valid_epochs, valid_metrics = parse_log_file(os.path.join(cfg.output_dir, f"{cfg.mode}.log"))
    plot_metrics(train_epochs, train_losses, valid_epochs, valid_metrics, cfg.output_dir)

    with open('./test_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
        model.eval()
        model.set_state_dict(paddle.load(f"{cfg.output_dir}/{cfg.model}_best.pdparams"))
        index = random.randint(0, int(len(val_data['X_list'])/cfg.DATA.sample_points) - 1)
        X_temp = val_data['X_list'][index * cfg.DATA.sample_points: (index+1) * cfg.DATA.sample_points]
        y_temp = val_data['y_list'][index * cfg.DATA.sample_points: (index+1) * cfg.DATA.sample_points]
        plot_predictions(model, X_temp, y_temp, val_data['y_mean'], val_data['y_std'], cfg.output_dir)


            

@paddle.no_grad()
def valid(cfg: DictConfig, test_loader, model):
    metric_fn = paddle.nn.MSELoss()
    meter = AverageMeterDict()
    
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        branch1 = x_batch[:, :cfg.MLPMODEL.branch1_in].astype(cfg.dtype) # Airfoil coord[0:400], Normalized inlet velocity[400, 401], sdf [402]
        trunk = x_batch[:, cfg.MLPMODEL.branch1_in:].astype(cfg.dtype) # Trunk coord [403, 404]
        Y_true = y_batch.astype(cfg.dtype) # Normalized Lables: Pressure/rho, velocity_x, velocity_y, viscosity_turbulence
        Y_pred = model({
            'branch1': branch1, 
            'trunk': trunk,
            })
        metric = metric_fn(Y_pred, Y_true)
        meter.update({"metric": metric})

    msg = "[Valid][Metric]"
    for k, v in meter.avg.items():
        msg += f"{float(v):.4e}({k}), "
    logging.info(msg)
    return meter.avg["metric"]


def test(cfg: DictConfig):
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, f"{cfg.mode}.log"),
        level=logging.INFO,
        format="%(ascttime)s:%(levelname)s:%(message)s",
    )
    # initialize model
    if cfg.model == "DeepONet":
        model = DeepONet(**cfg.MLPMODEL)
    elif cfg.model == "KANONet":
        model = kan.KANONet(**cfg.KANMODEL)
    
    param_dict = paddle.load(f"{cfg.checkpoint}")
    model.set_state_dict(param_dict)

    model.eval()
    
    logging.info(f"loaded model {cfg.model} from {cfg.checkpoint}")

    # Load and process data
    t1 = default_timer()
    
    _, _, test_invel_mean, test_invel_std, test_labels_mean, test_labels_std, X, Y = preprocess_data_airfrans(**cfg.DATA, train=False)
    
    t2 = default_timer()
    logging.info(f"Loading data took {t2 - t1:.2f} seconds.")

    logging.info(f"Start testing {cfg.model}...")
    # complete_field_prediction(test_invel_mean, test_invel_std, test_labels_mean, test_labels_std, model, )
    plot_predictions(model, X, Y, test_labels_mean, test_labels_std, cfg.output_dir)
    
    



@hydra.main(
    version_base=None, config_path="./conf", config_name="main.yaml"
)

def main(cfg: DictConfig):
    import hydra
    
    if cfg.seed is not None:
        set_seed(cfg.seed)
    if cfg.mode == "train":
        print("################## training #####################")
        train(cfg, with_val=True)
    elif cfg.mode == "test":
        print("################## test #####################")
        print("Load pretrained model")
        test(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'test'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

