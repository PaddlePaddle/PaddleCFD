import glob
import logging
import os
from timeit import default_timer
import time
from typing import List
import pickle
import random

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from paddle.distributed import fleet
from tqdm import tqdm
from scipy.io import loadmat

import matplotlib.pyplot as plt

from ppcfd.models.ppkan import kan_rbf
from mlp import DeepONet

def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)

def get_data(filename, ndata):
    r =15
    s = int(((421-1) / 15) + 1)

    data = loadmat(filename)
    x_branch = data["coeff"][:ndata, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, ::r, ::r].astype(np.float32) * 100

    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_branch = x_branch.reshape(ndata, s * s)

    y = y.reshape(ndata, s * s)
    return x_branch, grid, y

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
        model = kan_rbf.KANONet(**cfg.KANMODEL)
    if cfg.checkpoint:
        param_dict = paddle.load(f"{cfg.checkpoint}.pdparams")
        model.set_state_dict(param_dict)
    model.train()
    # initialize optimizer and scheduler
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=cfg.TRAIN.lr, weight_decay=1e-4
    )
    if cfg.enable_ddp:
        model = fleet.distributed(model)
        optimizer = fleet.distributed(optimizer)
        # load optimizer
    resume_ep = cfg.TRAIN.resume_ep

    if cfg.checkpoint and os.path.exists(f"{cfg.checkpoint}.pdopt"):
        optim_dict = paddle.load(f"{cfg.checkpoint}.pdopt")
        optimizer.set_state_dict(optim_dict)
        resume_ep = optim_dict["LR_Scheduler"]["last_epoch"]
    error_msg = (
        f"training epochs {cfg.TRAIN.epochs} should be greater than resume epoch, "
        f"which is {resume_ep} now."
    )
    assert cfg.TRAIN.epochs > resume_ep, error_msg

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=optimizer.get_lr(),
        T_max=cfg.TRAIN.epochs,
        last_epoch=resume_ep,
    )
    optimizer.set_lr_scheduler(scheduler)
    logging.info(f"lr of {resume_ep+1} is {optimizer.get_lr()}")

    # set loss_fn
    criterion = paddle.nn.MSELoss()
    metric = Relative_Error

    # Load and process data
    t1 = default_timer()
    c, x, y = get_data(cfg.data_dir, 1000)
    c_train = c[0:800]
    y_train = y[0:800]
    c_test = c[800:]
    y_test = y[800:]
    c_train = paddle.to_tensor(c_train, dtype=cfg.dtype)
    c_test = paddle.to_tensor(c_test, dtype=cfg.dtype)
    y_train = paddle.to_tensor(y_train, dtype=cfg.dtype)
    y_test = paddle.to_tensor(y_test, dtype=cfg.dtype)
    x = paddle.to_tensor(x, dtype=cfg.dtype)
    
    t2 = default_timer()
    logging.info(f"Loading data took {t2 - t1:.2f} seconds.")

    # training
    logging.info(f"Start training {cfg.model}...")
    train_losses = []
    test_losses = []
    start_time = time.time()
    mean_test_loss = 0.0
    results = []
    batch_size = cfg.TRAIN.batch_size

    epochs = cfg.TRAIN.epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = range(0, len(c_train), batch_size)
        
        progress_bar = tqdm(range(0, len(c_train), batch_size), desc=f'Epoch{epoch + 1}/{epochs}')
        
        for i in progress_bar:
            c_batch = c_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            

            optimizer.clear_grad()
            if cfg.model == 'KANONet':
                y_pred = model(c_batch, x)
            elif cfg.model == 'DeepONet':
                y_pred = model({
                    "branch1": c_batch,
                    "trunk": x,
                    })
            
            loss = criterion(y_pred, y_batch)
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            
            progress_bar.set_postfix({'Batch loss': loss.item()})
        avg_loss = total_loss / (len(c_train) // batch_size + 1)
        logging.info(f'Epoch {epoch+1}, Average Train Loss: {avg_loss:.4f}')
        train_losses.append(avg_loss)
        test_loss = test_model(model=model, criterion=criterion, c_test=c_test, y_test=y_test, x=x, batch_size=batch_size, cfg_model=cfg.model)
        test_losses.append(test_loss)
        results.append([epoch+1, avg_loss, test_loss])
        if epoch % 100 == 0:
            logging.info(f'Epoch {epoch}, Average Train Loss: {avg_loss}, Test Loss: {test_loss}')
            paddle.save(
                model.state_dict(), f"{cfg.output_dir}/{cfg.model}_latest.pdparams"
            )
            paddle.save(
                optimizer.state_dict(), f"{cfg.output_dir}/{cfg.model}_latest.pdopt"
            )
        scheduler.step()
    end_time = time.time()
    training_time = end_time - start_time

    idx = np.random.choice(c_test.shape[0], 1, replace=False)

    model.eval()
    with paddle.no_grad():
        if cfg.model == 'KANONet':
                y_pred = model(c_test[idx], x)
        elif cfg.model == 'DeepONet':
                y_pred = model({
                    "branch1": c_test[idx],
                    "trunk": x,
                    })
        y_pred = y_pred.reshape(shape=[29, 29]).numpy()
        y_true = y_test[idx].reshape(shape=[29, 29]).numpy()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Display the images
        im1 = axs[0].imshow(y_pred, cmap='jet', vmin=0, vmax=1)
        im2 = axs[1].imshow(y_true, cmap='jet', vmin=0, vmax=1)

        # Add a single color bar for both subplots
        cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

        plt.savefig(os.path.join(cfg.output_dir, f"sample_{idx}_result.png"))
        plt.close()

            

@paddle.no_grad()
def test_model(model, criterion, c_test, y_test, x, batch_size, cfg_model):
    model.eval()
    total_loss = 0
    num_batches = len(c_test) // batch_size + 1
    with paddle.no_grad():
        for i in range(0, len(c_test), batch_size):
            c_batch = c_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            if cfg_model == 'KANONet':
                pred = model(c_batch, x)
            elif cfg_model == 'DeepONet':
                pred = model({
                    "branch1": c_batch,
                    "trunk": x,
                    })
            # pred = model(c_batch, x)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
    loss_avg = total_loss / num_batches
    return loss_avg

def Relative_Error(y_true, y_pred):
    return paddle.mean(paddle.abs((y_pred - y_true))/(y_true))
    
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
        model = kan_rbf.KANONet(**cfg.KANMODEL)
    
    param_dict = paddle.load(f"{cfg.checkpoint}")
    model.set_state_dict(param_dict)

    model.eval()
    
    logging.info(f"loaded model {cfg.model} from {cfg.checkpoint}")
    # Load and process data
    
    t1 = default_timer()
    c, x, y = get_data(cfg.data_dir, 1000)
    c_test = c[800:]
    y_test = y[800:]
    c_test = paddle.to_tensor(c_test, dtype=cfg.dtype)
    y_test = paddle.to_tensor(y_test, dtype=cfg.dtype)
    x = paddle.to_tensor(x, dtype=cfg.dtype)
   
    t2 = default_timer()
    logging.info(f"Loading data took {t2 - t1:.2f} seconds.")

    logging.info(f"Start testing {cfg.model}...")

    test_loss = test_model(model=model, criterion=paddle.nn.MSELoss(), c_test=c_test, y_test=y_test, x=x, batch_size=cfg.TRAIN.batch_size, cfg_model=cfg.model)
    logging.info(f"Testing finished, average test loss: {test_loss:.6f}")

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
            f"cfg.mode should in ['train', 'valid', 'test'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

