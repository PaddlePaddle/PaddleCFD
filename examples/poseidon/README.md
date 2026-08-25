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

### Evaluation

Evaluate a trained checkpoint on the test set with `inference.py`. Metrics (loss and relative L1 error statistics) are appended to the CSV file given by `--file`:

```bash
cd examples/poseidon
python inference.py \
  --mode eval \
  --model_path <CHECKPOINT_PATH> \
  --dataset fluids.compressible.steady.Airfoil \
  --data_path <DATA_PATH> \
  --file eval_results.csv \
  --ckpt_dir <CHECKPOINT_DIR>
```

The `--dataset` must match the dataset the checkpoint was trained on (channel count and time conditioning). Use the `.time` suffix for time-conditioned checkpoints. Other modes (`save_samples`, `eval_accumulation_error`, `eval_resolutions`, ...) are available via `python inference.py --help`.

## CINN acceleration

`POSEIDON_USE_CINN=1` sets the three CINN FLAGS (`prim_enable_dynamic`, `prim_all`, `use_cinn`) before `import paddle` and wraps the model with `paddle.jit.to_static(model, full_graph=True)`. With the switch off (default), training falls back to the original pure dynamic-graph path.

`benchmark_compiler.py` times the two modes against each other on a synthetic batch, so no dataset download is needed. It runs one subprocess per mode (the CINN FLAGS must be set before `import paddle`) and prints a comparison:

```bash
cd examples/poseidon
python benchmark_compiler.py                        # both modes, ScOT-T, batch 32, 10 warm-up + 40 timed steps
python benchmark_compiler.py --model B --steps 60   # larger model, more steps
python benchmark_compiler.py --modes cinn           # single mode
python benchmark_compiler.py --time_conditioning    # time-dependent datasets (ConditionalLayerNorm)
```

The report gives median / mean / min / max ms per step for each mode, the steady-state speedup, the break-even step count (how many steps it takes to repay compilation), and a loss comparison between the two modes. A loss gap above 1% is flagged as a warning: `to_static` numerical divergence invalidates any speedup number.

Notes:

- Warm-up wall time under `cinn` is dominated by compilation, so `--warmup` must stay large enough for the timed window to be steady-state.
- Like `trainer.py`, the CINN path calls the model with `return_dict=False`: under `to_static(full_graph=True)` the `ScOTOutput` dataclass carries pir Values that cannot `backward()`.
- Measured this way on one H800 (ScOT-T, batch 4, 2 warm-up + 5 timed steps, synthetic SE-AF-shaped batch): 91.7 ms/step dynamic vs 54.8 ms/step CINN (**40.3% faster**), ~266 s one-time compilation, break-even around 7200 steps, and bit-identical losses between the two modes.

**Verified speedup** (ScOT-T, SE-AF, batch_size=32, steady-state per step, median of 40 steps after warm-up):

| Mode | Steady train step | Speedup |
| ---- | ----------------- | ------- |
| Dynamic graph (baseline) | 273 ms | — |
| CINN (to_static) | 196 ms | **28.1%** |

CINN pays a one-time compilation cost; short runs may be slower end-to-end, but steady-state training is ~28% faster, so the gain dominates over longer runs. Two datasets have been verified end-to-end (train + eval + test) under CINN: **SE-AF** (steady compressible flow) and **Poisson-Gauss** (elliptic Poisson equation).

## Datasets

### Getting the data

All datasets are published on the HuggingFace Hub, in two collections:

- [Pretraining datasets](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)
- [Downstream-task datasets](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14)

Download one with the Hub CLI and place it under the directory passed to `--data_path` (the launch scripts default to `examples/dataset`):

```bash
huggingface-cli download camlab-ethz/SE-AF --repo-type dataset --local-dir examples/dataset
```

### Code identifiers and dataset files

Datasets are HDF5 files (`.nc` / `.h5`). A dataset is selected by its **code identifier** in the YAML config (the `dataset:` field). The mapping to the Hub dataset names / local files:

| Code identifier | Dataset |
| ---------------- | ------- |
| `fluids.incompressible.Sines` | NS-Sines |
| `fluids.incompressible.Gaussians` | NS-Gauss |
| `fluids.incompressible.ShearLayer` | NS-SL |
| `fluids.incompressible.PiecewiseConstants` | NS-PwC |
| `fluids.incompressible.PiecewiseConstants.tracer` | NS-Tracer-PwC |
| `fluids.incompressible.VortexSheet` | NS-SVS |
| `fluids.incompressible.BrownianBridge` | NS-BB |
| `fluids.incompressible.forcing.KolmogorovFlow` | FNS-KF |
| `fluids.compressible.steady.Airfoil` | SE-AF |
| `fluids.compressible.Riemann` | CE-RP |
| `fluids.compressible.RiemannCurved` | CE-CRP |
| `fluids.compressible.RiemannKelvinHelmholtz` | CE-RPUI |
| `fluids.compressible.KelvinHelmholtz` | CE-KH |
| `fluids.compressible.Gaussians` | CE-Gauss |
| `fluids.compressible.RichtmyerMeshkov` | CE-RM |
| `fluids.compressible.gravity.RayleighTaylor` | GCE-RT |
| `elliptic.poisson.Gaussians` | Poisson-Gauss |
| `elliptic.Helmholtz` | Helmholtz |
| `wave.Layer` | Wave-Layer |
| `wave.Gaussians` | Wave-Gauss |
| `reaction_diffusion.AllenCahn` | ACE |

Two suffixes modify loading: appending `.time` loads a time-independent dataset as a time-dependent (long-time) one, and `.out` loads the out-of-distribution variant with more time steps.

### Verified in this project

| Dataset | Problem | Code identifier |
| ------- | ------- | --------------- |
| **SE-AF** | steady compressible flow (airfoil) | `fluids.compressible.steady.Airfoil` |
| **Poisson-Gauss** | elliptic Poisson equation | `elliptic.poisson.Gaussians` |

Both have been verified end-to-end (train + eval + test) on this codebase, in plain dynamic-graph mode and under CINN acceleration.

## Training configuration

Training is YAML-driven (`configs/run_small.yaml`). Key fields:

| Field | Meaning |
| ----- | ------- |
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
