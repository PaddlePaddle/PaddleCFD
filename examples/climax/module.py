from typing import Any

import paddle
from lr_scheduler import LinearWarmupCosineAnnealingLR
from metrics import (lat_weighted_mse_val, lat_weighted_nrmse,
                     lat_weighted_rmse, mse)
from normalize import Normalize
from ppcfd.models.climax.climatebench import ClimaXClimateBench
from ppcfd.models.climax.pos_embed import interpolate_pos_embed


class ClimateProjectionModule:
    """Plain paddle training module for climate projection with the ClimaXClimateBench model.

    Replaces the torch LightningModule: hyperparameters are stored directly on
    the instance (instead of ``save_hyperparameters``), logged metrics are
    appended to ``self.log_history`` (instead of ``self.log``) and consumed by
    the custom training loop in ``train.py``.

    Args:
        net (ClimaXClimateBench): ClimaXClimateBench model.
        pretrained_path (str, optional): Path to pre-trained checkpoint (pdparams).
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: ClimaXClimateBench,
        pretrained_path: str = "",
        lr: float = 0.0005,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-05,
        warmup_epochs: int = 60,
        max_epochs: int = 600,
        warmup_start_lr: float = 1e-08,
        eta_min: float = 1e-08,
    ):
        self.net = net
        # replaces save_hyperparameters(logger=False, ignore=["net"])
        self.pretrained_path = pretrained_path
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        # replaces LightningModule.log(): one list of floats per metric name
        self.log_history = {}
        # freeze_encoder (semantics of torch arch.py:69-76): all transformer
        # block params except norms are frozen via stop_gradient. Applied here
        # (after net construction) so it holds regardless of how the net was
        # built; idempotent with the net's own freezing.
        if getattr(net, "freeze_encoder", False):
            for name, p in net.blocks.named_parameters():
                name = name.lower()
                # we do not freeze the norm layers, as suggested by https://arxiv.org/abs/2103.05247
                if "norm" in name:
                    continue
                else:
                    p.stop_gradient = True
        if len(pretrained_path) > 0:
            self.load_mae_weights(pretrained_path)

    def load_mae_weights(self, pretrained_path):
        # pdparams checkpoint converted from the torch .ckpt:
        # {"state_dict": {keys carrying the "net." prefix}}
        checkpoint = paddle.load(path=str(pretrained_path))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding (expects "net."-prefixed keys)
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)
        # strip the lightning "net." prefix -> ClimaXClimateBench-level keys
        checkpoint_model = {
            k[len("net."):]: v
            for k, v in checkpoint_model.items()
            if k.startswith("net.")
        }
        state_dict = self.net.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
            if "token_embeds" in k or "head" in k:  # initialize embedding from scratch
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                continue
        for k in list(checkpoint_model.keys()):
            if (
                k not in state_dict.keys()
                or checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model (paddle set_state_dict is lenient by default,
        # equivalent to torch's strict=False; returns
        # (missing_keys, unexpected_keys))
        msg = self.net.set_state_dict(checkpoint_model)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        loss_dict, _ = self.net.forward(
            x, y, lead_times, variables, out_variables, [mse], lat=self.lat
        )
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log_history.setdefault("train/" + var, []).append(
                float(loss_dict[var])
            )
        loss = loss_dict["loss"]

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=None,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log_history.setdefault("val/" + var, []).append(
                float(loss_dict[var])
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_nrmse],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=None,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log_history.setdefault("test/" + var, []).append(
                float(loss_dict[var])
            )
        return loss_dict

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.net.named_parameters():
            if p.stop_gradient:
                continue  # frozen params (freeze_encoder) are excluded
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        opt = paddle.optimizer.AdamW(
            learning_rate=self.lr,
            beta1=self.beta_1,
            beta2=self.beta_2,
            epsilon=1e-8,
            parameters=[
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
        )
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            self.warmup_epochs,
            self.max_epochs,
            self.warmup_start_lr,
            self.eta_min,
        )
        return opt, sched
