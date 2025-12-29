import functools
import logging
import math
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from resample import UniformSampler

from ppcfd.models.confild import LossType
from ppcfd.models.confild import ModelMeanType
from ppcfd.models.confild import ModelVarType
from ppcfd.models.confild import SpacedDiffusion
from ppcfd.models.confild import UNetModel


def mean_flat(tensor):
    """Flatten mean over all dimensions except batch"""
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class Normalizer_ts(object):
    """Time-series data normalizer"""

    def __init__(self, params=None, method="-11", dim=None):
        self.params = params if params is not None else []
        self.method = method
        self.dim = dim

    def encode(self, data):
        """Normalize data"""
        if self.method == "-11":
            return 2 * (data - self.params[1]) / (self.params[0] - self.params[1]) - 1
        elif self.method == "01":
            return (data - self.params[1]) / (self.params[0] - self.params[1])
        return data

    def decode(self, data):
        """Denormalize data"""
        if self.method == "-11":
            return (data + 1) / 2 * (self.params[0] - self.params[1]) + self.params[1]
        elif self.method == "01":
            return data * (self.params[0] - self.params[1]) + self.params[1]
        return data

    def fit(self, data):
        """Fit normalizer to data"""
        if self.method in ["-11", "01"]:
            if self.dim is None:
                self.params = paddle.max(data), paddle.min(data)
            else:
                self.params = (
                    paddle.max(data, axis=self.dim, keepdim=True),
                    paddle.min(data, axis=self.dim, keepdim=True),
                )


def create_diffusion_model(cfg: DictConfig):
    """
    Create U-Net model for diffusion

    Args:
        cfg: Configuration object

    Returns:
        UNetModel instance
    """
    attention_resolutions = [int(x) for x in cfg.UNET.attention_resolutions.split(",")]

    model_params = {
        "image_size": cfg.UNET.image_size,
        "in_channels": cfg.UNET.in_channels,
        "model_channels": cfg.UNET.num_channels,
        "out_channels": cfg.UNET.out_channels,
        "num_res_blocks": cfg.UNET.num_res_blocks,
        "attention_resolutions": tuple(attention_resolutions),
        "dropout": 0,
        "channel_mult": tuple(cfg.UNET.channel_mult),
        "num_classes": None,
        "use_checkpoint": False,
        "use_fp16": False,
        "num_heads": cfg.UNET.num_heads,
        "num_head_channels": cfg.UNET.num_head_channels,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "dims": 2,
    }

    model = UNetModel(**model_params)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def create_diffusion(cfg: DictConfig):
    """
    Create diffusion process

    Args:
        cfg: Configuration object

    Returns:
        SpacedDiffusion instance
    """
    # Get beta schedule
    betas = get_named_beta_schedule(cfg.DIFF.noise_schedule, cfg.DIFF.steps)

    diffusion = SpacedDiffusion(
        use_timesteps=range(cfg.DIFF.steps),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,  # Fixed variance (not learned)
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    print(f"Diffusion model created with {cfg.DIFF.steps} steps, noise schedule: {cfg.DIFF.noise_schedule}")

    return diffusion


def prepare_data(cfg: DictConfig):
    """
    Load and prepare latent data for training

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (train_data, val_data, normalizer)
    """
    # Load training and validation data
    train_data = np.load(cfg.DATA.train_data)
    valid_data = np.load(cfg.DATA.valid_data)

    print(f"Train data shape: {train_data.shape}, range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    print(f"Valid data shape: {valid_data.shape}, range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")

    # Convert to tensors
    train_tensor = paddle.to_tensor(train_data, dtype="float32")
    valid_tensor = paddle.to_tensor(valid_data, dtype="float32")

    # Reshape 3D data to 4D: [N, D, D] -> [N, 1, D, D]
    # This converts latent codes to image-like format: [batch, channels=1, height, width]
    if train_tensor.ndim == 3:
        train_tensor = train_tensor.unsqueeze(axis=1)  # Add channel dimension at axis 1
        valid_tensor = valid_tensor.unsqueeze(axis=1)

    # Normalize data
    normalizer = Normalizer_ts(method="-11")
    normalizer.fit(train_tensor)

    train_normalized = normalizer.encode(train_tensor)
    valid_normalized = normalizer.encode(valid_tensor)

    print(
        f"After normalization: train range: [{float(train_normalized.min()):.3f}, {float(train_normalized.max()):.3f}]"
    )

    logging.info(f"Train data shape: {train_normalized.shape}")
    logging.info(f"Val data shape: {valid_normalized.shape}")

    return train_normalized, valid_normalized, normalizer


def train(cfg: DictConfig, with_val=True):
    """
    Train diffusion model

    Args:
        cfg: Configuration object
        with_val: Whether to perform validation during training
    """
    # Logging is already configured in main()

    # Prepare data
    train_data, val_data, normalizer = prepare_data(cfg)

    # Create models
    unet_model = create_diffusion_model(cfg)
    diffusion = create_diffusion(cfg)

    # Load checkpoint if specified
    if cfg.checkpoint:
        unet_model.set_state_dict(paddle.load(cfg.checkpoint))
        logging.info(f"Loaded checkpoint from {cfg.checkpoint}")

    # Create optimizer
    optimizer = paddle.optimizer.AdamW(
        learning_rate=cfg.TRAIN.lr, parameters=unet_model.parameters(), weight_decay=cfg.TRAIN.get("weight_decay", 0.0)
    )
    print(f"Optimizer initialized with lr={cfg.TRAIN.lr}, weight_decay={cfg.TRAIN.get('weight_decay', 0.0)}")

    # Create EMA model
    ema_params = [param.clone().detach() for param in unet_model.parameters()]
    # Ensure EMA params don't track gradients
    for ema_param in ema_params:
        ema_param.stop_gradient = True
    ema_rate = float(cfg.TRAIN.ema_rate)

    # Training setup
    schedule_sampler = UniformSampler(diffusion)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print(f"Starting training: max_steps={cfg.TRAIN.max_steps}, lr_anneal_steps={cfg.TRAIN.get('lr_anneal_steps', 0)}")

    for step in range(1, cfg.TRAIN.max_steps + 1):
        unet_model.train()

        # Sample batch
        batch_size = cfg.TRAIN.batch_size
        indices = np.random.randint(0, len(train_data), size=batch_size)
        batch = train_data[indices]

        # Sample timesteps
        t, weights = schedule_sampler.sample(batch_size)

        # Compute losses
        compute_losses = functools.partial(
            diffusion.training_losses,
            unet_model,
            batch,
            t,
        )

        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()

        # Backward pass
        optimizer.clear_grad()
        loss.backward()

        # Gradient clipping for stability
        paddle.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update EMA
        for ema_param, param in zip(ema_params, unet_model.parameters()):
            with paddle.no_grad():
                updated = ema_rate * ema_param + (1 - ema_rate) * param
                ema_param.set_value(updated)

        # Learning rate annealing
        if cfg.TRAIN.get("lr_anneal_steps", 0) > 0:
            final_lr = cfg.TRAIN.get("final_lr", 0.0)
            frac_done = min(step / cfg.TRAIN.lr_anneal_steps, 1.0)
            # Linear annealing: from lr to final_lr
            new_lr = cfg.TRAIN.lr * (1 - frac_done) + final_lr * frac_done
            optimizer.set_lr(new_lr)

        # Logging
        if step % cfg.log_freq == 0:
            train_losses.append(loss.item())
            current_lr = optimizer.get_lr()
            logging.info(f"Step {step}/{cfg.TRAIN.max_steps} - Loss: {loss.item():.6f}, LR: {current_lr:.2e}")
            print(f"Step {step}: Loss={loss.item():.6f}, LR={current_lr:.2e}")

        # Validation
        save_interval = cfg.TRAIN.get("save_interval", 1000)
        if with_val and (step % save_interval == 0 or step == cfg.TRAIN.max_steps):
            val_loss = valid(cfg, val_data, unet_model, diffusion, schedule_sampler)
            val_losses.append(val_loss)

            logging.info(f"Step {step} - Val Loss: {val_loss:.6f}")
            print(f"Step {step}: Validation Loss={val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                paddle.save(unet_model.state_dict(), os.path.join(cfg.output_dir, "unet_best.pdparams"))

                # Save EMA parameters
                ema_state_dict = {
                    name: ema_param
                    for name, ema_param in zip([n for n, _ in unet_model.named_parameters()], ema_params)
                }
                paddle.save(ema_state_dict, os.path.join(cfg.output_dir, "ema_best.pdparams"))
                logging.info(f"Saved best model with val loss: {val_loss:.6f}")

        # Periodic save
        if step % save_interval == 0:
            paddle.save(unet_model.state_dict(), os.path.join(cfg.output_dir, f"unet_{step}.pdparams"))

    # Save final models
    paddle.save(unet_model.state_dict(), os.path.join(cfg.output_dir, "unet_final.pdparams"))

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    if train_losses:
        train_steps = [i * cfg.log_freq for i in range(1, len(train_losses) + 1)]
        plt.plot(train_steps, train_losses, label="Train Loss", alpha=0.7)
    if val_losses:
        val_steps = [i * cfg.TRAIN.save_interval for i in range(1, len(val_losses) + 1)]
        plt.plot(val_steps, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Diffusion Model Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg.output_dir, "diffusion_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logging.info("Training completed!")


@paddle.no_grad()
def valid(cfg: DictConfig, val_data, unet_model, diffusion, schedule_sampler):
    """
    Validate diffusion model

    Args:
        cfg: Configuration object
        val_data: Validation data
        unet_model: U-Net model
        diffusion: Diffusion process
        schedule_sampler: Timestep sampler

    Returns:
        Average validation loss
    """
    unet_model.eval()

    val_loss_accum = 0.0
    num_batches = 0

    for i in range(0, len(val_data), cfg.TRAIN.test_batch_size):
        batch = val_data[i : i + cfg.TRAIN.test_batch_size]
        batch_size = len(batch)

        # Sample timesteps (don't use weights in validation for unbiased evaluation)
        t, _ = schedule_sampler.sample(batch_size)

        # Compute losses without importance sampling weights
        losses = diffusion.training_losses(unet_model, batch, t)
        loss = losses["loss"].mean()  # Unweighted mean for fair validation

        val_loss_accum += loss.item()
        num_batches += 1

    return val_loss_accum / num_batches


def test(cfg: DictConfig):
    """
    Test diffusion model by generating samples

    Args:
        cfg: Configuration object
    """
    # Logging is already configured in main()

    # Prepare data
    _, val_data, normalizer = prepare_data(cfg)

    # Create models
    unet_model = create_diffusion_model(cfg)
    diffusion = create_diffusion(cfg)

    # Load best model
    model_path = cfg.checkpoint if cfg.checkpoint else os.path.join(cfg.output_dir, "ema_best.pdparams")
    unet_model.set_state_dict(paddle.load(model_path))
    logging.info(f"Loaded model from {model_path}")

    unet_model.eval()

    # Generate samples
    num_samples = min(cfg.TRAIN.test_batch_size, len(val_data))
    # Shape should be [batch, in_channels, time_length, latent_length]
    time_length = cfg.EVAL.time_length if hasattr(cfg, "EVAL") else 128
    latent_length = cfg.EVAL.latent_length if hasattr(cfg, "EVAL") else 128
    shape = [num_samples, cfg.UNET.in_channels, time_length, latent_length]

    logging.info(f"Generating {num_samples} samples with shape {shape}")

    with paddle.no_grad():
        # Sample from diffusion model
        samples = diffusion.p_sample_loop(
            unet_model,
            shape,
            clip_denoised=True,
            model_kwargs={},
        )

        # Denormalize
        samples_denorm = normalizer.decode(samples)

        # Save results
        output_path = os.path.join(cfg.output_dir, "generated_samples.npy")
        np.save(output_path, samples_denorm.numpy())
        logging.info(f"Saved generated samples to {output_path}")

        # Compare with real data
        real_samples = val_data[:num_samples]
        real_denorm = normalizer.decode(real_samples)

        mse = paddle.nn.functional.mse_loss(samples_denorm, real_denorm)
        mae = paddle.abs(samples_denorm - real_denorm).mean()

        logging.info(f"Generation MSE: {mse.item():.6f}")
        logging.info(f"Generation MAE: {mae.item():.6f}")
        print(f"Generation MSE: {mse.item():.6f}, MAE: {mae.item():.6f}")


@hydra.main(version_base=None, config_path="./conf", config_name="un_confild_case1.yaml")
def main(cfg: DictConfig):
    """Main entry point"""
    paddle.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Setup logging once for the entire program
    log_file = os.path.join(cfg.output_dir, f"{cfg.mode}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # Python 3.8+ allows reconfiguration
    )

    if cfg.mode == "train":
        print("################## Training Diffusion Model #####################")
        train(cfg, with_val=True)
    elif cfg.mode == "test":
        print("################## Testing Diffusion Model #####################")
        test(cfg)
    else:
        raise ValueError(f"cfg.mode should be 'train' or 'test', but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
