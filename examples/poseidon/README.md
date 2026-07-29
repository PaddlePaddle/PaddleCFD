# Poseidon (ScOT) — A PDE Foundation Model

A scalable operator transformer (ScOT) for solving partial differential equations (PDEs), built on a Swin Transformer V2 U-Net with ConvNeXt residual skip connections. It targets a broad family of physics problems — incompressible / compressible fluids, elliptic equations, wave propagation, and reaction-diffusion systems — under a single trained operator.

## Model architecture

| Component | Description |
|-----------|-------------|
| Backbone | Swin Transformer V2 encoder–decoder U-Net |
| Skip connections | ConvNeXt / ResNet residual blocks |
| Conditioning | Optional time conditioning via `ConditionalLayerNorm` |
| I/O | Patch embedding (input) and patch recovery (output) |

The model is configured through `ScOTConfig`. Four widths are available:

| Scale | `depths` | `embed_dim` | Params (approx.) |
|-------|----------|-------------|------------------|
| T (Tiny)  | [4, 4, 4, 4]  | 48  | ~20M |
| S (Small) | [8, 8, 8, 8]  | 48  | ~30M |
| B (Base)  | [8, 8, 8, 8]  | 96  | ~90M |
| L (Large) | [8, 8, 8, 8]  | 192 | ~300M |

```python
from ppcfd.models.poseidon import ScOT, ScOTConfig

config = ScOTConfig(
    image_size=128,
    patch_size=4,
    num_channels=1,
    num_out_channels=1,
    embed_dim=48,
    depths=[4, 4, 4, 4],
    num_heads=[3, 6, 12, 24],
    skip_connections=[2, 2, 2, 0],
    window_size=16,
)
model = ScOT(config)

# Or load a local checkpoint
model = ScOT.from_pretrained("/path/to/poseidon_checkpoint")
```

## Pretrained checkpoint

The PaddlePaddle Tiny checkpoint (ScOT-T, ~20M params) is converted from the original pretrained weights and hosted on AIStudio:

> https://aistudio.baidu.com/modelsdetail/49243/intro

Download it, then load with:

```python
from ppcfd.models.poseidon import ScOT
model = ScOT.from_pretrained("/path/to/downloaded/poseidon_tiny_paddle")
```

## Quick start

### Plain training (dynamic graph)

```bash
cd examples/poseidon
bash minimal_train.sh
```

### CINN accelerated training (to_static)

The PaddlePaddle CINN compiler only takes effect after `paddle.jit.to_static`. A single switch `POSEIDON_USE_CINN` controls both the CINN-related FLAGS and the `to_static` wrapping, so CINN is engaged with no code change:

```bash
cd examples/poseidon
bash minimal_train_CINN.sh          # equivalent to: POSEIDON_USE_CINN=1 python train.py ...
```

### Direct invocation

```bash
cd examples/poseidon
python train.py \
  --config configs/run_small.yaml \
  --wandb_run_name "se-af-scratch-small" \
  --wandb_project_name "PaddleCFD-Poseidon" \
  --checkpoint_path <CHECKPOINT_PATH> \
  --data_path <DATA_PATH>
```

Use `python train.py --help` for the full argument list. `wandb` can be disabled via `WANDB_MODE=disabled`, but the `wandb` package must still be installed (the script imports it).

## CINN acceleration

`POSEIDON_USE_CINN=1` sets the three CINN FLAGS (`prim_enable_dynamic`, `prim_all`, `use_cinn`) before `import paddle` and wraps the model with `paddle.jit.to_static(model, full_graph=True)`. With the switch off (default), training falls back to the original pure dynamic-graph path.

**Verified speedup** (ScOT-T, SE-AF, batch_size=32, steady-state per step, median of 40 steps after warm-up):

| Mode | Steady train step | Speedup |
|------|-------------------|---------|
| Dynamic graph (baseline) | 273 ms | — |
| CINN (to_static) | 196 ms | **28.1%** |

CINN pays a one-time compilation cost; short runs may be slower end-to-end, but steady-state training is ~28% faster, so the gain dominates over longer runs. Two datasets have been verified end-to-end (train + eval + test) under CINN: **SE-AF** (steady compressible flow) and **Poisson-Gauss** (elliptic Poisson equation).

## Datasets

Datasets are HDF5 files (`.nc` / `.h5`) placed at the path given by `--data_path`. Supported problem types:

| Category               | Problems                     |
| ---------------------- | ---------------------------- |
| **Fluids**             | incompressible, compressible (steady/transient) |
| **Elliptic**           | Poisson, Helmholtz           |
| **Wave**               | acoustic                     |
| **Reaction-Diffusion** | Allen-Cahn                   |

The dataset is selected by its code identifier in the YAML config (e.g. `fluids.compressible.steady.Airfoil`, `elliptic.poisson.Gaussians`).

## Training configuration

Training is YAML-driven (`configs/run_small.yaml`). Key fields:

| Field | Meaning |
|-------|---------|
| `dataset` | Dataset code identifier |
| `num_trajectories` | Number of training trajectories (`-1` for full set) |
| `model_name` | Model scale: `T` / `S` / `B` / `L` |
| `batch_size` | Per-device batch size |
| `num_epochs` | Total training epochs |
| `lr` / `lr_scheduler` | Learning rate and schedule (`cosine`, `linear`, `constant`) |
| `early_stopping_patience` | Early-stop patience on eval loss |

## Additional dependencies

Beyond core PaddleCFD, the Poseidon example scripts require:

```bash
python -m pip install psutil wandb
```
