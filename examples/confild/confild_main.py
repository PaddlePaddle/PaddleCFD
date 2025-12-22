import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.io import Dataset
from paddle.io import DistributedBatchSampler

from ppcfd.models.confild import LatentContainer
from ppcfd.models.confild import SIRENAutodecoder_film


def get_model_keys(cfg, model_type="LATENT"):
    """
    Get input/output keys from config or use defaults.

    Args:
        cfg: Configuration object
        model_type: Model configuration section name ('LATENT' or 'CONFILD')

    Returns:
        tuple: (input_key, output_key)
    """
    config = getattr(cfg, model_type)
    input_key = config.input_keys[0] if hasattr(config, "input_keys") else "input"
    output_key = config.output_keys[0] if hasattr(config, "output_keys") else "output"
    return input_key, output_key


# Dataset class for distributed training
class CoNFILDDataset(Dataset):
    """
    Dataset for distributed training with DataLoader workers.

    NOTE: This class is ONLY used in train_distributed(), not in train().
    Single-GPU training uses direct tensor indexing without DataLoader.

    IMPORTANT: fois_data and coords must be NumPy arrays, not GPU tensors,
    because DataLoader workers cannot access GPU memory from the main process.
    """

    def __init__(self, fois_data, coords, indices, global_offset=0):
        """
        Args:
            fois_data: Flow field data as NumPy array (local slice)
            coords: Coordinates as NumPy array (can be None)
            indices: Local indices for this split (relative to fois_data)
            global_offset: Offset to convert local indices to global indices for LatentContainer
        """
        self.fois_data = fois_data
        self.coords = coords
        self.indices = indices
        self.global_offset = global_offset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Local index within this dataset split
        local_idx = self.indices[idx]
        fois = self.fois_data[local_idx]

        # Convert NumPy to tensor in worker process (creates tensor in correct CUDA context)
        # The isinstance check is defensive programming - currently fois is always NumPy
        if isinstance(fois, np.ndarray):
            fois = paddle.to_tensor(fois, dtype="float32")

        # Global index for LatentContainer lookup
        global_idx = local_idx + self.global_offset

        return {
            "fois": fois,
            "sample_idx": global_idx,  # Return global index
        }


# Data loading functions
def load_elbow_flow(path):
    """Load elbow flow data"""
    return np.load(f"{path}")[1:]


def load_channel_flow(path, t_start=0, t_end=1200, t_every=1):
    """Load channel flow data"""
    return np.load(f"{path}")[t_start:t_end:t_every]


def load_periodic_hill_flow(path):
    """Load periodic hill flow data"""
    return np.load(f"{path}")


def load_3d_flow(path):
    """Load 3D flow data"""
    return np.load(f"{path}")


def rMAE(prediction, target, dims=(1, 2)):
    """Relative Mean Absolute Error"""
    return paddle.abs(prediction - target).mean(axis=dims) / paddle.abs(target).mean(axis=dims)


class Normalizer_ts(object):
    """Time-series data normalizer"""

    def __init__(self, params=[], method="-11", dim=None):
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
        """Fit normalizer and normalize data"""
        assert isinstance(data, paddle.Tensor)
        if len(self.params) == 0:
            if self.method in ["-11", "01"]:
                if self.dim is None:
                    self.params = paddle.max(data), paddle.min(data)
                else:
                    max_val = paddle.max(data, axis=self.dim, keepdim=True)
                    min_val = paddle.min(data, axis=self.dim, keepdim=True)
                    self.params = max_val, min_val
            elif self.method == "ms":
                if self.dim is None:
                    self.params = paddle.mean(data), paddle.std(data)
                else:
                    self.params = (
                        paddle.mean(data, axis=self.dim, keepdim=True),
                        paddle.std(data, axis=self.dim, keepdim=True),
                    )
            elif self.method == "none":
                return data
        return self.fnormalize(data, self.params, self.method)

    def normalize(self, new_data):
        """Normalize new data using fitted parameters"""
        if new_data.place != self.params[0].place:
            # Move params to the same device as new_data (avoid hardcoding cuda)
            device = new_data.place
            self.params = (
                paddle.to_tensor(self.params[0].numpy(), place=device),
                paddle.to_tensor(self.params[1].numpy(), place=device),
            )
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data):
        """Denormalize data"""
        if new_data.place != self.params[0].place:
            # Move params to the same device as new_data (avoid hardcoding cuda)
            device = new_data.place
            self.params = (
                paddle.to_tensor(self.params[0].numpy(), place=device),
                paddle.to_tensor(self.params[1].numpy(), place=device),
            )
        return self.fdenormalize(new_data, self.params, self.method)

    @staticmethod
    def fnormalize(data, params, method):
        """Static normalization function"""
        if method == "-11":
            return 2 * (data - params[1]) / (params[0] - params[1]) - 1
        elif method == "01":
            return (data - params[1]) / (params[0] - params[1])
        elif method == "ms":
            return (data - params[0]) / params[1]
        elif method == "none":
            return data

    @staticmethod
    def fdenormalize(data, params, method):
        """Static denormalization function"""
        if method == "-11":
            return (data + 1) / 2 * (params[0] - params[1]) + params[1]
        elif method == "01":
            return data * (params[0] - params[1]) + params[1]
        elif method == "ms":
            return data * params[1] + params[0]
        elif method == "none":
            return data


def prepare_data(cfg: DictConfig):
    """
    Load and prepare data for training/validation/testing

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (normed_coords, normed_fois, normalizer, spatial_axis, train_size)
    """
    # Load data based on configuration
    load_fn = globals()[cfg.DATA.load_data_fn]
    fois = load_fn(cfg.DATA.data_path)

    # Load coordinates
    coor_path = cfg.DATA.get("coor_path", None)
    if coor_path:
        coords = np.load(coor_path)
    else:
        coords = None
        logging.warning("No coordinate path provided, using default")

    # Convert to tensors
    fois_tensor = paddle.to_tensor(fois, dtype="float32")

    # Normalize data
    normalizer = Normalizer_ts(method=cfg.DATA.normalizer.method, dim=cfg.DATA.normalizer.dim)
    normed_fois = normalizer.fit_normalize(fois_tensor)

    # Normalize coordinates and extract spatial dimensions
    spatial_axis = fois.shape[1:-1]  # Always extract spatial dimensions

    if coords is not None:
        coords_tensor = paddle.to_tensor(coords, dtype="float32")
        spatial_normalizer = Normalizer_ts(method="-11")
        normed_coords = spatial_normalizer.fit_normalize(coords_tensor)
    else:
        normed_coords = None

    # Split into train/val/test (70/15/15)
    total_samples = normed_fois.shape[0]
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)

    logging.info(f"Total samples: {total_samples}")
    logging.info(f"Train: {train_size}, Val: {val_size}, Test: {total_samples - train_size - val_size}")
    logging.info(f"Spatial dimensions: {spatial_axis}")

    return normed_coords, normed_fois, normalizer, spatial_axis, train_size


def train_distributed(cfg: DictConfig, with_val=True):
    """
    Distributed training for CNF model using Fleet

    Args:
        cfg: Configuration object
        with_val: Whether to perform validation during training
    """
    # Initialize Fleet for distributed training
    fleet.init(is_collective=True)

    # Setup logging - only rank 0 writes to file to avoid conflicts
    rank = fleet.worker_index()
    world_size = fleet.worker_num()

    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(cfg.output_dir, "train.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Distributed training started on {world_size} GPUs")
    else:
        # Other ranks: only log warnings to console
        logging.basicConfig(
            level=logging.WARNING,
            format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        )

    # Get model keys for LatentContainer
    latent_input_key, latent_output_key = get_model_keys(cfg, "LATENT")

    # Prepare data
    normed_coords, normed_fois, normalizer, spatial_axis, train_size = prepare_data(cfg)

    # Convert to NumPy for DataLoader workers
    # Workers run in separate CPU processes and cannot access GPU tensors
    # This conversion happens once before training starts
    train_fois = normed_fois[:train_size].numpy()
    val_fois = normed_fois[train_size:].numpy()

    # Convert coordinates to numpy if present
    normed_coords_np = normed_coords.numpy() if normed_coords is not None else None

    # Create dataset indices (local indices within each split)
    train_indices = list(range(len(train_fois)))
    val_indices = list(range(len(val_fois)))

    # Create datasets with global offset for LatentContainer
    # Train set: global indices [0, train_size)
    # Val set: global indices [train_size, train_size+val_size)
    train_dataset = CoNFILDDataset(train_fois, normed_coords_np, train_indices, global_offset=0)
    val_dataset = CoNFILDDataset(val_fois, normed_coords_np, val_indices, global_offset=train_size)

    # Create distributed samplers
    train_sampler = DistributedBatchSampler(
        train_dataset, batch_size=cfg.TRAIN.batch_size, shuffle=True, drop_last=True
    )
    val_sampler = DistributedBatchSampler(
        val_dataset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False, drop_last=False
    )

    # Create data loaders
    num_workers = cfg.TRAIN.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=num_workers, use_shared_memory=False
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers, use_shared_memory=False)

    # Restore coords as tensor for model use (reused across batches)
    # The coords tensor is used directly by the model, not per-sample in Dataset
    normed_coords = paddle.to_tensor(normed_coords_np, dtype="float32") if normed_coords_np is not None else None

    # Initialize models
    confild = SIRENAutodecoder_film(**cfg.CONFILD)
    latent = LatentContainer(**cfg.LATENT)

    # Wrap models with Fleet
    confild = fleet.distributed_model(confild)
    latent = fleet.distributed_model(latent)

    # Load checkpoint if specified
    if cfg.checkpoint:
        confild.set_state_dict(paddle.load(f"{cfg.checkpoint}_confild.pdparams"))
        latent.set_state_dict(paddle.load(f"{cfg.checkpoint}_latent.pdparams"))
        logging.info(f"Loaded checkpoint from {cfg.checkpoint}")

    # Create optimizers
    cnf_optimizer = paddle.optimizer.Adam(
        learning_rate=cfg.TRAIN.lr.cnf, parameters=confild.parameters(), weight_decay=0.0
    )
    latents_optimizer = paddle.optimizer.Adam(
        learning_rate=cfg.TRAIN.lr.latents, parameters=latent.parameters(), weight_decay=0.0
    )

    # Wrap optimizers with Fleet
    cnf_optimizer = fleet.distributed_optimizer(cnf_optimizer)
    latents_optimizer = fleet.distributed_optimizer(latents_optimizer)

    # Loss function
    loss_fn = paddle.nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(1, cfg.TRAIN.epochs + 1):
        confild.train()
        latent.train()

        train_loss_accum = 0.0
        num_batches = 0

        # Training
        for batch in train_loader:
            batch_fois = batch["fois"]
            batch_sample_idx = batch["sample_idx"]

            # Forward pass - use actual sample indices for latent lookup
            latent_z = latent({latent_input_key: batch_sample_idx})[latent_output_key]

            if normed_coords is not None:
                batch_size = len(batch_fois)
                predictions = confild(
                    {"confild_x": normed_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
                )["confild_output"]
            else:
                # For structured grids without explicit coordinates
                batch_size = batch_fois.shape[0]
                spatial_dims = batch_fois.shape[1:-1]

                # Create grid coordinates on the fly
                # IMPORTANT: Use indexing='ij' for consistent coordinate ordering
                # This ensures coordinates match the data layout: (H, W) -> [(0,0), (0,1), ..., (H-1, W-1)]
                coord_arrays = [paddle.linspace(-1, 1, dim) for dim in spatial_dims]
                try:
                    # Paddle 3.x: try with indexing parameter
                    mesh_grids = paddle.meshgrid(*coord_arrays, indexing="ij")
                except TypeError:
                    # Paddle 2.x: no indexing parameter
                    mesh_grids = paddle.meshgrid(*coord_arrays)

                grid_coords = paddle.stack(mesh_grids, axis=-1)
                grid_coords = grid_coords.reshape([-1, len(spatial_dims)])

                predictions = confild(
                    {"confild_x": grid_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
                )["confild_output"]

                # Reshape predictions to match batch_fois
                predictions = predictions.reshape([batch_size] + list(spatial_dims) + [-1])

            loss = loss_fn(predictions, batch_fois)

            # Backward pass
            loss.backward()
            cnf_optimizer.step()
            latents_optimizer.step()
            cnf_optimizer.clear_grad()
            latents_optimizer.clear_grad()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches if num_batches > 0 else 0

        # Synchronize training loss across all ranks for accurate global average
        train_loss_tensor = paddle.to_tensor([avg_train_loss], dtype="float32")
        paddle.distributed.all_reduce(train_loss_tensor)
        avg_train_loss = train_loss_tensor.item() / fleet.worker_num()

        train_losses.append(avg_train_loss)

        # Validation - all ranks participate, then synchronize results
        if with_val and (epoch % cfg.log_freq == 0 or epoch == cfg.TRAIN.epochs):
            # All ranks run validation on their data slice
            val_loss = valid_distributed(cfg, val_loader, normed_coords, confild, latent, loss_fn)

            # Synchronize validation loss across all ranks
            val_loss_tensor = paddle.to_tensor([val_loss], dtype="float32")
            paddle.distributed.all_reduce(val_loss_tensor)
            val_loss = val_loss_tensor.item() / fleet.worker_num()

            # Only rank 0 logs and saves models
            if fleet.worker_index() == 0:
                val_losses.append(val_loss)
                logging.info(
                    f"Epoch {epoch}/{cfg.TRAIN.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    paddle.save(confild.state_dict(), os.path.join(cfg.output_dir, "cnf_model_best.pdparams"))
                    paddle.save(latent.state_dict(), os.path.join(cfg.output_dir, "latents_model_best.pdparams"))
                    logging.info(f"Saved best model with val loss: {val_loss:.6f}")
        else:
            if fleet.worker_index() == 0 and epoch % cfg.log_freq == 0:
                logging.info(f"Epoch {epoch}/{cfg.TRAIN.epochs} - Train Loss: {avg_train_loss:.6f}")

            # Periodic save
            if epoch % 1000 == 0:
                paddle.save(confild.state_dict(), os.path.join(cfg.output_dir, f"cnf_model_{epoch}.pdparams"))
                paddle.save(latent.state_dict(), os.path.join(cfg.output_dir, f"latents_model_{epoch}.pdparams"))

    # Save final models (only on rank 0)
    if fleet.worker_index() == 0:
        paddle.save(confild.state_dict(), os.path.join(cfg.output_dir, "cnf_model_final.pdparams"))
        paddle.save(latent.state_dict(), os.path.join(cfg.output_dir, "latents_model_final.pdparams"))

        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        if with_val and val_losses:
            val_epochs = [i * cfg.log_freq for i in range(1, len(val_losses) + 1)]
            plt.plot(val_epochs, val_losses, label="Val Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("CNF Distributed Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(cfg.output_dir, "cnf_loss_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

        logging.info("Distributed training completed!")


@paddle.no_grad()
def valid_distributed(cfg: DictConfig, val_loader, normed_coords, confild, latent, loss_fn):
    """
    Validate CNF model in distributed mode

    Args:
        cfg: Configuration object
        val_loader: Validation data loader
        normed_coords: Normalized coordinates
        confild: CNF model
        latent: Latent container
        loss_fn: Loss function

    Returns:
        Average validation loss
    """
    # Get model keys
    latent_input_key, latent_output_key = get_model_keys(cfg, "LATENT")

    confild.eval()
    latent.eval()

    val_loss_accum = 0.0
    num_batches = 0

    for batch in val_loader:
        batch_fois = batch["fois"]
        batch_sample_idx = batch["sample_idx"]

        # Get latent codes for validation samples
        latent_z = latent({latent_input_key: batch_sample_idx})[latent_output_key]

        if normed_coords is not None:
            batch_size = len(batch_fois)
            predictions = confild(
                {"confild_x": normed_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
            )["confild_output"]
        else:
            # For structured grids without explicit coordinates
            batch_size = batch_fois.shape[0]
            spatial_dims = batch_fois.shape[1:-1]

            # Create grid coordinates (consistent with training)
            coord_arrays = [paddle.linspace(-1, 1, dim) for dim in spatial_dims]
            try:
                # Paddle 3.x: try with indexing parameter
                mesh_grids = paddle.meshgrid(*coord_arrays, indexing="ij")
            except TypeError:
                # Paddle 2.x: no indexing parameter
                mesh_grids = paddle.meshgrid(*coord_arrays)

            grid_coords = paddle.stack(mesh_grids, axis=-1).reshape([-1, len(spatial_dims)])

            predictions = confild(
                {"confild_x": grid_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
            )["confild_output"]
            predictions = predictions.reshape([batch_size] + list(spatial_dims) + [-1])

        loss = loss_fn(predictions, batch_fois)
        val_loss_accum += loss.item()
        num_batches += 1

    return val_loss_accum / num_batches if num_batches > 0 else 0


def train(cfg: DictConfig, with_val=True):
    """
    Train CNF model

    Args:
        cfg: Configuration object
        with_val: Whether to perform validation during training
    """
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Get model keys for LatentContainer
    latent_input_key, latent_output_key = get_model_keys(cfg, "LATENT")

    # Prepare data
    normed_coords, normed_fois, normalizer, spatial_axis, train_size = prepare_data(cfg)

    # Split data
    train_fois = normed_fois[:train_size]
    val_fois = normed_fois[train_size:]

    # Initialize models
    confild = SIRENAutodecoder_film(**cfg.CONFILD)
    latent = LatentContainer(**cfg.LATENT)

    # Load checkpoint if specified
    if cfg.checkpoint:
        confild.set_state_dict(paddle.load(f"{cfg.checkpoint}_confild.pdparams"))
        latent.set_state_dict(paddle.load(f"{cfg.checkpoint}_latent.pdparams"))
        logging.info(f"Loaded checkpoint from {cfg.checkpoint}")

    # Create optimizers
    cnf_optimizer = paddle.optimizer.Adam(
        learning_rate=cfg.TRAIN.lr.cnf, parameters=confild.parameters(), weight_decay=0.0
    )
    latents_optimizer = paddle.optimizer.Adam(
        learning_rate=cfg.TRAIN.lr.latents, parameters=latent.parameters(), weight_decay=0.0
    )

    # Loss function
    loss_fn = paddle.nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(1, cfg.TRAIN.epochs + 1):
        confild.train()
        latent.train()

        train_loss_accum = 0.0
        num_batches = 0

        # Training
        indices = np.random.permutation(train_size)
        for i in range(0, train_size, cfg.TRAIN.batch_size):
            batch_indices = indices[i : i + cfg.TRAIN.batch_size]
            batch_data = train_fois[batch_indices]

            # Forward pass - use actual batch indices for latent lookup
            latent_z = latent({latent_input_key: paddle.to_tensor(batch_indices, dtype="int64")})[latent_output_key]

            if normed_coords is not None:
                coords_input = normed_coords.unsqueeze(0).tile([len(batch_indices), 1, 1])
                predictions = confild({"confild_x": coords_input, "latent_z": latent_z})["confild_output"]
            else:
                # For structured grids without explicit coordinates
                batch_size = batch_data.shape[0]
                spatial_dims = batch_data.shape[1:-1]

                # Create grid coordinates on the fly
                coord_arrays = [paddle.linspace(-1, 1, dim) for dim in spatial_dims]
                try:
                    mesh_grids = paddle.meshgrid(*coord_arrays, indexing="ij")
                except TypeError:
                    mesh_grids = paddle.meshgrid(*coord_arrays)

                grid_coords = paddle.stack(mesh_grids, axis=-1)
                grid_coords = grid_coords.reshape([-1, len(spatial_dims)])

                coords_input = grid_coords.unsqueeze(0).tile([batch_size, 1, 1])

                predictions = confild({"confild_x": coords_input, "latent_z": latent_z})["confild_output"]

                # Reshape predictions to match batch_data
                predictions = predictions.reshape([batch_size] + list(spatial_dims) + [-1])

            loss = loss_fn(predictions, batch_data)

            # Backward pass
            loss.backward()
            cnf_optimizer.step()
            latents_optimizer.step()
            cnf_optimizer.clear_grad()
            latents_optimizer.clear_grad()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        if with_val and (epoch % cfg.log_freq == 0 or epoch == cfg.TRAIN.epochs):
            val_loss = valid(cfg, val_fois, normed_coords, confild, latent, loss_fn, train_size)
            val_losses.append(val_loss)

            logging.info(
                f"Epoch {epoch}/{cfg.TRAIN.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                paddle.save(confild.state_dict(), os.path.join(cfg.output_dir, "cnf_model_best.pdparams"))
                paddle.save(latent.state_dict(), os.path.join(cfg.output_dir, "latents_model_best.pdparams"))
                logging.info(f"Saved best model with val loss: {val_loss:.6f}")
        else:
            if epoch % cfg.log_freq == 0:
                logging.info(f"Epoch {epoch}/{cfg.TRAIN.epochs} - Train Loss: {avg_train_loss:.6f}")

        # Periodic save
        if epoch % 1000 == 0:
            paddle.save(confild.state_dict(), os.path.join(cfg.output_dir, f"cnf_model_{epoch}.pdparams"))
            paddle.save(latent.state_dict(), os.path.join(cfg.output_dir, f"latents_model_{epoch}.pdparams"))

    # Save final models
    paddle.save(confild.state_dict(), os.path.join(cfg.output_dir, "cnf_model_final.pdparams"))
    paddle.save(latent.state_dict(), os.path.join(cfg.output_dir, "latents_model_final.pdparams"))

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    if with_val and val_losses:
        val_epochs = [i * cfg.log_freq for i in range(1, len(val_losses) + 1)]
        plt.plot(val_epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CNF Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.output_dir, "cnf_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logging.info("Training completed!")


@paddle.no_grad()
def valid(cfg: DictConfig, val_fois, normed_coords, confild, latent, loss_fn, train_size):
    """
    Validate CNF model

    Args:
        cfg: Configuration object
        val_fois: Validation data
        normed_coords: Normalized coordinates
        confild: CNF model
        latent: Latent container
        loss_fn: Loss function
        train_size: Size of training set (for latent indexing)

    Returns:
        Average validation loss
    """
    # Get model keys
    latent_input_key, latent_output_key = get_model_keys(cfg, "LATENT")

    confild.eval()
    latent.eval()

    val_loss_accum = 0.0
    num_batches = 0
    val_size = val_fois.shape[0]

    for i in range(0, val_size, cfg.TRAIN.test_batch_size):
        batch_data = val_fois[i : i + cfg.TRAIN.test_batch_size]
        batch_size = batch_data.shape[0]

        # Get latent codes for validation samples - correct indexing
        latent_start_idx = train_size + i
        latent_indices = paddle.arange(latent_start_idx, latent_start_idx + batch_size, dtype="int64")
        latent_z = latent({latent_input_key: latent_indices})[latent_output_key]

        if normed_coords is not None:
            predictions = confild(
                {"confild_x": normed_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
            )["confild_output"]
        else:
            # For structured grids without explicit coordinates
            spatial_dims = batch_data.shape[1:-1]

            # Create grid coordinates (consistent with training)
            coord_arrays = [paddle.linspace(-1, 1, dim) for dim in spatial_dims]
            try:
                # Paddle 3.x: try with indexing parameter
                mesh_grids = paddle.meshgrid(*coord_arrays, indexing="ij")
            except TypeError:
                # Paddle 2.x: no indexing parameter
                mesh_grids = paddle.meshgrid(*coord_arrays)

            grid_coords = paddle.stack(mesh_grids, axis=-1).reshape([-1, len(spatial_dims)])

            predictions = confild(
                {"confild_x": grid_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
            )["confild_output"]
            predictions = predictions.reshape([batch_size] + list(spatial_dims) + [-1])

        loss = loss_fn(predictions, batch_data)
        val_loss_accum += loss.item()
        num_batches += 1

    return val_loss_accum / num_batches


def test(cfg: DictConfig):
    """
    Test CNF model

    Args:
        cfg: Configuration object
    """
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, "test.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Get model keys for LatentContainer
    latent_input_key, latent_output_key = get_model_keys(cfg, "LATENT")

    # Prepare data
    normed_coords, normed_fois, normalizer, spatial_axis, train_size = prepare_data(cfg)

    # Use test split
    test_fois = normed_fois[train_size + int(normed_fois.shape[0] * 0.15) :]
    test_size = test_fois.shape[0]

    # Initialize models
    confild = SIRENAutodecoder_film(**cfg.CONFILD)
    latent = LatentContainer(**cfg.LATENT)

    # Load best models
    if cfg.checkpoint:
        # If checkpoint is provided, assume it's a prefix (e.g., "path/to/model")
        confild_path = f"{cfg.checkpoint}_confild.pdparams"
        latent_path = f"{cfg.checkpoint}_latent.pdparams"
    else:
        # Use best models from output directory
        confild_path = os.path.join(cfg.output_dir, "cnf_model_best.pdparams")
        latent_path = os.path.join(cfg.output_dir, "latents_model_best.pdparams")

    confild.set_state_dict(paddle.load(confild_path))
    latent.set_state_dict(paddle.load(latent_path))
    logging.info(f"Loaded models from {confild_path} and {latent_path}")

    confild.eval()
    latent.eval()

    # Test
    test_loss_accum = 0.0
    test_errors = []
    num_batches = 0

    with paddle.no_grad():
        for i in range(0, test_size, cfg.TRAIN.test_batch_size):
            batch_data = test_fois[i : i + cfg.TRAIN.test_batch_size]
            batch_size = batch_data.shape[0]

            # Get latent codes - correct indexing for test set
            val_size = int(normed_fois.shape[0] * 0.15)
            latent_start = train_size + val_size + i
            latent_indices = paddle.arange(latent_start, latent_start + batch_size, dtype="int64")
            latent_z = latent({latent_input_key: latent_indices})[latent_output_key]

            if normed_coords is not None:
                predictions = confild(
                    {"confild_x": normed_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
                )["confild_output"]
            else:
                # For structured grids without explicit coordinates
                spatial_dims = batch_data.shape[1:-1]

                # Create grid coordinates (consistent with training)
                coord_arrays = [paddle.linspace(-1, 1, dim) for dim in spatial_dims]
                try:
                    # Paddle 3.x: try with indexing parameter
                    mesh_grids = paddle.meshgrid(*coord_arrays, indexing="ij")
                except TypeError:
                    # Paddle 2.x: no indexing parameter
                    mesh_grids = paddle.meshgrid(*coord_arrays)

                grid_coords = paddle.stack(mesh_grids, axis=-1).reshape([-1, len(spatial_dims)])

                predictions = confild(
                    {"confild_x": grid_coords.unsqueeze(0).tile([batch_size, 1, 1]), "latent_z": latent_z}
                )["confild_output"]
                predictions = predictions.reshape([batch_size] + list(spatial_dims) + [-1])

            # Denormalize for error calculation
            pred_denorm = normalizer.denormalize(predictions)
            target_denorm = normalizer.denormalize(batch_data)

            # Calculate errors
            mse = paddle.nn.functional.mse_loss(pred_denorm, target_denorm)
            mae = paddle.abs(pred_denorm - target_denorm).mean()

            test_loss_accum += mse.item()
            test_errors.append(mae.item())
            num_batches += 1

    avg_test_mse = test_loss_accum / num_batches
    avg_test_mae = np.mean(test_errors)

    logging.info(f"Test MSE: {avg_test_mse:.6f}")
    logging.info(f"Test MAE: {avg_test_mae:.6f}")
    print(f"Test MSE: {avg_test_mse:.6f}, Test MAE: {avg_test_mae:.6f}")


def check_gpu_availability(cfg: DictConfig):
    """
    Check GPU availability and validate configuration before training.

    Args:
        cfg: Configuration object

    Returns:
        tuple: (use_distributed, available_gpus, warnings)
    """
    import paddle

    warnings = []
    num_gpus_config = cfg.TRAIN.get("num_gpus", 1)
    available_gpus = paddle.device.cuda.device_count()

    print(f"GPU Check: {available_gpus} available, {num_gpus_config} configured")

    # Check CUDA availability
    if not paddle.device.is_compiled_with_cuda():
        print("Warning: Paddle not compiled with CUDA, using CPU")
        return False, 0, warnings

    # Check if any GPU is available
    if available_gpus == 0:
        print("Warning: No GPU detected, using CPU")
        return False, 0, warnings

    # Check if configured GPUs exceed available GPUs
    if num_gpus_config > available_gpus:
        print(f"Error: Configured {num_gpus_config} GPUs but only {available_gpus} available")
        print(f"Fix: Set TRAIN.num_gpus={available_gpus}")
        raise RuntimeError(f"GPU mismatch: requested {num_gpus_config}, available {available_gpus}")

    # Determine if distributed training should be used
    use_distributed = num_gpus_config > 1

    if use_distributed:
        print(f"Using {num_gpus_config} GPUs (effective batch size: {cfg.TRAIN.batch_size * num_gpus_config})")

        # Check if launched with paddle.distributed.launch
        import os

        if "PADDLE_TRAINER_ID" not in os.environ:
            print("Warning: Use 'python -m paddle.distributed.launch --gpus 0,1 ...' for multi-GPU training")
    else:
        print(f"Using single GPU (batch size: {cfg.TRAIN.batch_size})")

    return use_distributed, available_gpus, warnings


@hydra.main(version_base=None, config_path="./conf", config_name="confild_main.yaml")
def main(cfg: DictConfig):
    """Main entry point"""
    paddle.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.mode == "train":
        # Check GPU availability before training
        use_distributed, available_gpus, warnings = check_gpu_availability(cfg)

        if use_distributed:
            print("################## Distributed Training CNF Model #####################")
            import paddle.distributed as dist

            dist.init_parallel_env()
            train_distributed(cfg, with_val=True)
        else:
            print("################## Training CNF Model #####################")
            train(cfg, with_val=True)
    elif cfg.mode == "test":
        print("################## Testing CNF Model #####################")
        test(cfg)
    else:
        raise ValueError(f"cfg.mode should be 'train' or 'test', but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
