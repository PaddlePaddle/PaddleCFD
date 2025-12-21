import logging
import os
from timeit import default_timer

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from simulate import SimulationDataset

# Add parent directory to path for ppcfd imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from ppcfd.models.symbolic_gn import HGN, OGN, VarOGN, get_edge_index


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    paddle.seed(seed)
    np.random.seed(seed)


def create_batch_edge_index(edge_index, batch_size, num_nodes):
    """
    Create correct batch edge index by adjusting node index offsets

    Args:
        edge_index: Original edge index [2, num_edges]
        batch_size: Batch size
        num_nodes: Number of nodes per graph

    Returns:
        Batch edge index [2, batch_size * num_edges]
    """
    batch_edges = []
    for b in range(batch_size):
        offset = b * num_nodes
        batch_edges.append(edge_index + offset)
    return np.concatenate(batch_edges, axis=1)


def create_model(cfg: DictConfig, edge_index: np.ndarray):
    """
    Create model based on configuration

    Args:
        cfg: Hydra configuration object
        edge_index: Graph edge index

    Returns:
        Model instance
    """
    model_type = cfg.MODEL.arch
    n_f = cfg.DATA.dimension * 2 + 2  # [x, y, vx, vy, charge, mass] for 2D
    ndim = cfg.DATA.dimension

    l1_strength = 0.0
    if cfg.MODEL.regularization_type == "l1" and model_type in ["OGN", "VarOGN"]:
        l1_strength = cfg.MODEL.l1_strength

    if model_type == "OGN":
        model = OGN(
            n_f=n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=ndim,
            hidden=cfg.MODEL.hidden,
            edge_index=edge_index,
            l1_strength=l1_strength,
        )
    elif model_type == "HGN":
        model = HGN(
            n_f=n_f,
            ndim=ndim,
            hidden=cfg.MODEL.hidden,
            edge_index=edge_index,
        )
    elif model_type == "VarOGN":
        model = VarOGN(
            n_f=n_f,
            msg_dim=cfg.MODEL.msg_dim,
            ndim=ndim,
            hidden=cfg.MODEL.hidden,
            edge_index=edge_index,
            l1_strength=l1_strength,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def prepare_data(cfg: DictConfig):
    """
    Generate and prepare simulation data

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, edge_index)
    """
    logging.info(f"Generating {cfg.DATA.type} simulation data...")
    t1 = default_timer()

    sim = SimulationDataset(
        sim=cfg.DATA.type,
        n=cfg.DATA.num_nodes,
        dim=cfg.DATA.dimension,
        nt=cfg.DATA.time_steps,
        dt=cfg.DATA.time_step_size,
    )
    sim.simulate(cfg.DATA.num_samples)
    accel_data = sim.get_acceleration()

    t2 = default_timer()
    logging.info(f"Data generation took {t2 - t1:.2f} seconds.")

    # Prepare training data
    X_list = []
    y_list = []
    for sample_idx in range(cfg.DATA.num_samples):
        for t in range(0, sim.data.shape[1], cfg.DATA.sample_interval):
            X_list.append(sim.data[sample_idx, t])
            y_list.append(accel_data[sample_idx, t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Split data with shuffling to avoid temporal bias
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    logging.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create edge index
    edge_index = get_edge_index(cfg.DATA.num_nodes, cfg.DATA.type)

    return X_train, y_train, X_val, y_val, X_test, y_test, edge_index


def train(cfg: DictConfig, with_val=True):
    """
    Train the model

    Args:
        cfg: Configuration object
        with_val: Whether to perform validation during training
    """
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    set_seed(cfg.seed)

    # Prepare data
    X_train, y_train, X_val, y_val, _, _, edge_index = prepare_data(cfg)

    # Create model
    model = create_model(cfg, edge_index)
    if cfg.checkpoint:
        checkpoint_path = cfg.checkpoint if cfg.checkpoint.endswith('.pdparams') else f"{cfg.checkpoint}.pdparams"
        model.set_state_dict(paddle.load(checkpoint_path))
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    model.train()

    # Create optimizer and learning rate scheduler
    if cfg.TRAIN.lr_scheduler.name == "CosineAnnealingDecay":
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=cfg.TRAIN.optimizer.learning_rate,
            T_max=cfg.TRAIN.epochs
        )
    elif cfg.TRAIN.lr_scheduler.name == "ExponentialDecay":
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=cfg.TRAIN.optimizer.learning_rate,
            gamma=cfg.TRAIN.lr_scheduler.get("gamma", 0.95),
            # ExponentialDecay in PaddlePaddle uses epoch-based decay
            # No need for decay_steps parameter in epoch-based scheduling
        )
    else:
        lr_scheduler = cfg.TRAIN.optimizer.learning_rate

    optimizer = paddle.optimizer.Adam(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=cfg.TRAIN.optimizer.weight_decay,
    )

    # Create loss function
    if cfg.TRAIN.loss.type == "MAE":
        loss_fn = paddle.nn.L1Loss(reduction="mean")
    else:
        loss_fn = paddle.nn.MSELoss(reduction="mean")

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, cfg.TRAIN.epochs + 1):
        # model.train() is redundant here since valid() switches back to train mode
        train_loss_accum = 0.0
        num_batches = 0

        # Shuffle training data each epoch
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), cfg.TRAIN.batch_size):
            batch_indices = indices[i:i + cfg.TRAIN.batch_size]
            batch_x = paddle.to_tensor(X_train[batch_indices], dtype="float32")
            batch_y = paddle.to_tensor(y_train[batch_indices], dtype="float32")
            # Use original edge_index, not batch edge_index
            # The model handles batching internally
            batch_edge = paddle.to_tensor(edge_index, dtype="int64")

            input_dict = {"x": batch_x, "edge_index": batch_edge}
            outputs = model(input_dict)
            pred_accel = outputs["acceleration"]

            loss = loss_fn(pred_accel, batch_y)

            # Add L1 regularization if present
            if "l1_regularization" in outputs:
                loss = loss + outputs["l1_regularization"]

            # Clear gradients before backward
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches
        train_losses.append(avg_train_loss)

        # Learning rate scheduling (per epoch)
        if hasattr(lr_scheduler, 'step'):
            lr_scheduler.step()

        # Validation
        if with_val and (epoch % cfg.log_freq == 0 or epoch == cfg.TRAIN.epochs):
            val_loss = valid(cfg, X_val, y_val, edge_index, model, loss_fn)
            val_losses.append(val_loss)

            logging.info(f"Epoch {epoch}/{cfg.TRAIN.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                paddle.save(model.state_dict(), os.path.join(cfg.output_dir, f"{cfg.MODEL.arch}_best.pdparams"))
                logging.info(f"Saved best model with val loss: {val_loss:.6f}")
        else:
            if epoch % cfg.log_freq == 0:
                logging.info(f"Epoch {epoch}/{cfg.TRAIN.epochs} - Train Loss: {avg_train_loss:.6f}")

        # Periodic save
        if epoch % cfg.TRAIN.save_freq == 0:
            paddle.save(model.state_dict(), os.path.join(cfg.output_dir, f"{cfg.MODEL.arch}_epoch_{epoch}.pdparams"))
            paddle.save(optimizer.state_dict(), os.path.join(cfg.output_dir, f"{cfg.MODEL.arch}_epoch_{epoch}.pdopt"))

    # Save final model
    paddle.save(model.state_dict(), os.path.join(cfg.output_dir, f"{cfg.MODEL.arch}_final.pdparams"))

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if with_val and val_losses:
        val_epochs = [i * cfg.log_freq for i in range(1, len(val_losses) + 1)]
        plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{cfg.MODEL.arch} Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.output_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    logging.info("Training completed!")


@paddle.no_grad()
def valid(cfg: DictConfig, X_val, y_val, edge_index, model, loss_fn):
    """
    Validate the model

    Args:
        cfg: Configuration object
        X_val: Validation input data
        y_val: Validation target data
        edge_index: Graph edge index
        model: Model to validate
        loss_fn: Loss function

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss_accum = 0.0
    num_batches = 0

    for i in range(0, len(X_val), cfg.TRAIN.batch_size):
        batch_x = paddle.to_tensor(X_val[i:i + cfg.TRAIN.batch_size], dtype="float32")
        batch_y = paddle.to_tensor(y_val[i:i + cfg.TRAIN.batch_size], dtype="float32")
        # Use original edge_index, not batch edge_index
        batch_edge = paddle.to_tensor(edge_index, dtype="int64")

        input_dict = {"x": batch_x, "edge_index": batch_edge}
        outputs = model(input_dict)
        pred_accel = outputs["acceleration"]

        # Validation loss should only include data fitting term, not regularization
        loss = loss_fn(pred_accel, batch_y)

        val_loss_accum += loss.item()
        num_batches += 1

    avg_val_loss = val_loss_accum / num_batches

    # Switch back to training mode
    model.train()

    return avg_val_loss


def test(cfg: DictConfig):
    """
    Test the model on test set

    Args:
        cfg: Configuration object
    """
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, "test.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    set_seed(cfg.seed)

    # Prepare data
    _, _, _, _, X_test, y_test, edge_index = prepare_data(cfg)

    # Load model
    model = create_model(cfg, edge_index)
    checkpoint_path = cfg.checkpoint if cfg.checkpoint else os.path.join(cfg.output_dir, f"{cfg.MODEL.arch}_best.pdparams")
    model.set_state_dict(paddle.load(checkpoint_path))
    logging.info(f"Loaded model from {checkpoint_path}")
    model.eval()

    # Create loss function
    if cfg.TRAIN.loss.type == "MAE":
        loss_fn = paddle.nn.L1Loss(reduction="mean")
    else:
        loss_fn = paddle.nn.MSELoss(reduction="mean")

    # Test
    test_loss_accum = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    with paddle.no_grad():
        for i in range(0, len(X_test), cfg.TRAIN.batch_size):
            batch_x = paddle.to_tensor(X_test[i:i + cfg.TRAIN.batch_size], dtype="float32")
            batch_y = paddle.to_tensor(y_test[i:i + cfg.TRAIN.batch_size], dtype="float32")
            # Use original edge_index, not batch edge_index
            batch_edge = paddle.to_tensor(edge_index, dtype="int64")

            input_dict = {"x": batch_x, "edge_index": batch_edge}
            outputs = model(input_dict)
            pred_accel = outputs["acceleration"]

            # Test loss should only include data fitting term, not regularization
            loss = loss_fn(pred_accel, batch_y)

            test_loss_accum += loss.item()
            num_batches += 1

            all_preds.append(pred_accel.numpy())
            all_targets.append(batch_y.numpy())

    avg_test_loss = test_loss_accum / num_batches
    logging.info(f"Test Loss ({cfg.TRAIN.loss.type}): {avg_test_loss:.6f}")
    print(f"Test Loss ({cfg.TRAIN.loss.type}): {avg_test_loss:.6f}")

    # Calculate additional metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = np.mean(np.abs(all_preds - all_targets))
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)

    logging.info(f"MAE: {mae:.6f}")
    logging.info(f"MSE: {mse:.6f}")
    logging.info(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")


@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def main(cfg: DictConfig):
    """Main entry point"""
    if cfg.seed is not None:
        set_seed(cfg.seed)

    if cfg.mode == "train":
        print("################## Training #####################")
        train(cfg, with_val=True)
    elif cfg.mode == "test":
        print("################## Testing #####################")
        test(cfg)
    else:
        raise ValueError(f"cfg.mode should be 'train' or 'test', but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
