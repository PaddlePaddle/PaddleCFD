# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-command training entry: `python train.py` (cwd = examples/climax).

All knobs live in configs/climate_projection.yaml; there is no CLI surface.
Set CLIMAX_USE_CINN=1 to train under to_static+CINN (paddle 3.x compiles
to_static graphs with CINN once wrapped).
"""

import os
import random
import statistics
import time

# CINN switch: a single env var gates both to_static and the CINN FLAGS
# (paddle 3.x binds them: to_static always compiles with CINN once wrapped,
# so separate switches would leave dead branches). Must run before any
# `import paddle` for the FLAGS to take effect.
_USE_CINN = os.environ.get("CLIMAX_USE_CINN", "0") == "1"
if _USE_CINN:
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
    os.environ.setdefault("FLAGS_print_ir", "false")
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"

import numpy as np  # noqa: E402
import paddle  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from datamodule import ClimateBenchDataModule  # noqa: E402
from module import ClimateProjectionModule  # noqa: E402
from ppcfd.models.climax.climatebench import ClimaXClimateBench  # noqa: E402

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "climate_projection.yaml"
)


def _epoch_means(module, prefix, start_lengths):
    """Mean over the log_history entries appended since start_lengths was taken."""
    means = {}
    for name, values in module.log_history.items():
        if not name.startswith(prefix):
            continue
        seg = values[start_lengths.get(name, 0):]
        if seg:
            means[name] = sum(seg) / len(seg)
    return means


def _print_metric_table(metrics, header="Test metric", source="DataLoader 0"):
    """Print a lightning-style two-column metric table (torch baseline format).

    Matches the torch baseline table: two columns of equal width, no leading
    empty column, cells centered with format-spec ``^`` (odd padding puts the
    extra space on the right, like lightning). Width is floored at 27 so the
    current metric key set renders exactly at the baseline's 27/27; longer
    keys/values widen the columns.
    """
    longest = max(
        [len(header), len(source)] + [len(k) for k in metrics] +
        [len(str(v)) for v in metrics.values()]
    )
    width = max(27, longest + 2)
    print("┏" + "━" * width + "┳" + "━" * width + "┓")
    print(f"┃{header:^{width}}┃{source:^{width}}┃")
    print("┡" + "━" * width + "╇" + "━" * width + "┩")
    for name in sorted(metrics.keys()):
        print(f"│{name:^{width}}│{metrics[name]:^{width}}│")
    print("└" + "─" * width + "┴" + "─" * width + "┘")


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    root_dir = cfg.train.default_root_dir
    os.makedirs(root_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(root_dir, "config.yaml"))

    datamodule = ClimateBenchDataModule(
        root_dir=cfg.data.root_dir,
        history=cfg.data.history,
        list_train_simu=list(cfg.data.list_train_simu),
        list_test_simu=list(cfg.data.list_test_simu),
        variables=list(cfg.data.variables),
        out_variables=cfg.data.out_variables,
        train_ratio=cfg.data.train_ratio,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    net = ClimaXClimateBench(**cfg.model.net.init_args)

    # model.init_state_path: load the FULL initial state (backbone + heads)
    # directly, bypassing load_mae_weights (whose cleaning drops token_embeds/
    # head keys and would leave the heads freshly initialized instead of shared
    # with the torch baseline).
    init_state_path = cfg.model.get("init_state_path", "")
    if init_state_path:
        ckpt = paddle.load(init_state_path)["state_dict"]
        full = {k[len("net."):]: v for k, v in ckpt.items() if k.startswith("net.")}
        missing, unexpected = net.set_state_dict(full)
        print(
            "Loaded full initial state from %s: missing=%s unexpected=%s"
            % (init_state_path, sorted(missing), sorted(unexpected))
        )
        if missing or unexpected:
            raise RuntimeError("strict initial state load failed")

    # CINN path: wrap with to_static AFTER the weights are loaded (parameters
    # are shared, so the later-built optimizer stays valid) and BEFORE the
    # module is constructed (module.net then refers to the wrapped object).
    if _USE_CINN:
        net = paddle.jit.to_static(net, full_graph=True)
        print("[CINN] to_static enabled, full_graph=True")
    module = ClimateProjectionModule(
        net=net,
        pretrained_path="" if init_state_path else cfg.model.get("pretrained_path", ""),
        lr=cfg.model.get("lr", 5e-4),
        beta_1=cfg.model.get("beta_1", 0.9),
        beta_2=cfg.model.get("beta_2", 0.99),
        weight_decay=cfg.model.get("weight_decay", 1e-5),
        warmup_epochs=cfg.model.get("warmup_epochs", 60),
        max_epochs=cfg.model.get("max_epochs", 600),
        warmup_start_lr=cfg.model.get("warmup_start_lr", 1e-8),
        eta_min=cfg.model.get("eta_min", 1e-8),
    )

    # assembly sequence replicated from torch main()
    normalization = datamodule.dataset_train.out_transform
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    module.set_denormalization(mean_denorm, std_denorm)
    module.set_lat_lon(*datamodule.get_lat_lon())
    module.set_pred_range(0)
    module.set_val_clim(None)
    module.set_test_clim(datamodule.get_test_clim())

    optimizer, lr_scheduler = module.configure_optimizers()

    max_epochs = cfg.train.get("max_epochs", 50)
    patience = int(cfg.train.get("patience", 5))
    monitor_key = "val/w_mse"

    best_path = os.path.join(root_dir, "best.pdparams")
    last_path = os.path.join(root_dir, "last.pdparams")
    best_metric = float("inf")
    bad_epochs = 0

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for epoch in range(max_epochs):
        module.net.train()
        # Per-step timing (identical in dynamic and CINN modes): wall clock
        # around train step + backward + optimizer/scheduler update.
        step_times = []
        epoch_train_start = time.perf_counter()
        for i, batch in enumerate(train_loader):
            step_start = time.perf_counter()
            loss = module.training_step(batch, i)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()
            step_times.append(time.perf_counter() - step_start)
            if (i + 1) % 50 == 0:
                cum = sum(step_times)
                print(
                    f"Epoch {epoch} step {i + 1}: train/loss = {float(loss):.4f}"
                    f" | cum={cum:.2f}s avg={cum / len(step_times):.4f}s"
                )
        # pure-training wall clock for this epoch (validation excluded)
        train_wall = time.perf_counter() - epoch_train_start
        steady = step_times[50:]  # skip warmup steps for the steady median
        steady_median = (
            f"{statistics.median(steady):.4f}s" if steady else "n/a"
        )
        print(
            f"[timing] epoch={epoch} train_wall={train_wall:.2f}s "
            f"steps={len(step_times)} steady_median={steady_median} "
            f"(excl. first 50 steps)"
        )

        # validation: epoch means of the metrics logged by validation_step
        module.net.eval()
        start_lengths = {
            name: len(vals)
            for name, vals in module.log_history.items()
            if name.startswith("val/")
        }
        with paddle.no_grad():
            for i, batch in enumerate(val_loader):
                module.validation_step(batch, i)
        val_metrics = _epoch_means(module, "val/", start_lengths)
        print(
            f"Epoch {epoch}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
        )

        # checkpoints: best on monitor + last at every epoch end (torch yaml:
        # ModelCheckpoint monitor=val/w_mse mode=min save_last=True)
        paddle.save(module.net.state_dict(), last_path)
        monitor = val_metrics.get(monitor_key)
        if monitor is None:
            raise RuntimeError(
                f"Monitored metric {monitor_key} missing from validation logs"
            )
        if monitor < best_metric:
            best_metric = monitor
            bad_epochs = 0
            paddle.save(module.net.state_dict(), best_path)
            print(f"Epoch {epoch}: {monitor_key} improved to {monitor:.6f}")
        else:
            bad_epochs += 1
            print(
                f"Epoch {epoch}: no {monitor_key} improvement for "
                f"{bad_epochs}/{patience} epochs"
            )
            # EarlyStopping (torch yaml: patience=5, min_delta=0., mode=min)
            if bad_epochs >= patience:
                print("Early stopping triggered")
                break

    # test the trained model with the best checkpoint (torch main():
    # ckpt_path='best')
    if os.path.exists(best_path):
        print(f"Restoring states from the checkpoint path at {best_path}")
        module.net.set_state_dict(paddle.load(best_path))
    else:
        print(f"No checkpoint found at {best_path}, testing current weights")

    module.net.eval()
    test_loader = datamodule.test_dataloader()
    start_lengths = {
        name: len(vals)
        for name, vals in module.log_history.items()
        if name.startswith("test/")
    }
    with paddle.no_grad():
        for i, batch in enumerate(test_loader):
            module.test_step(batch, i)
    test_metrics = _epoch_means(module, "test/", start_lengths)
    _print_metric_table(test_metrics)
    return test_metrics


if __name__ == "__main__":
    main()
