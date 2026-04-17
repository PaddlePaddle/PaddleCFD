# Poseidon (ScOT) in PaddleCFD

## Model Overview

Poseidon / ScOT (Scientific Operator Transformer) is a foundation model for solving partial differential equations (PDEs). It is built on the Swin Transformer V2 architecture and supports a wide range of physics problems including fluid dynamics, elliptic equations, wave propagation, and reaction-diffusion systems.

## Importing the Model

Import from PaddleCFD and configure with `ScOTConfig`:

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

# Or load a converted local checkpoint
model = ScOT.from_pretrained("/path/to/poseidon_paddle_checkpoint")
```

## Training

### Quick start

```bash
cd examples/poseidon
bash minimal_train.sh
```

### Direct Python invocation

```bash
cd examples/poseidon
python train.py \
  --config configs/run_small.yaml \
  --wandb_run_name "se-af-scratch-small" \
  --wandb_project_name "PaddleCFD-Poseidon" \
  --checkpoint_path <CHECKPOINT_PATH> \
  --data_path <DATA_PATH>
```

> **Note:** `wandb` can be disabled by setting `WANDB_MODE=disabled`, but the `wandb` package must still be installed because the training script imports it.

For all parameters, use:

```bash
cd examples/poseidon
python ./train.py --help
```

## Dataset

Datasets must be provided in HDF5 format and placed at the path configured via `--data_path`. The following problem types are supported:

| Category               | Problems                     |
| ---------------------- | ---------------------------- |
| **Fluids**             | incompressible, compressible |
| **Elliptic**           | poisson, helmholtz           |
| **Wave**               | acoustic                     |
| **Reaction-Diffusion** | allen-cahn                   |

See the original [Poseidon repository](https://arxiv.org/abs/2405.19101) for dataset download instructions and naming conventions.

## Checkpoint Conversion

Convert PyTorch checkpoints to PaddlePaddle format:

```bash
cd examples/poseidon
python convert_torch_ckpt_to_paddle.py --src /path/to/torch_checkpoint --dst /path/to/paddle_checkpoint
```

**Required extra dependencies for conversion:**

```bash
python -m pip install torch safetensors
```

After conversion, load the checkpoint:

```python
from ppcfd.models.poseidon import ScOT
model = ScOT.from_pretrained("/path/to/paddle_checkpoint")
```

## Additional Dependencies

The Poseidon example scripts require the following packages beyond core PaddleCFD dependencies:

```text
- psutil
- wandb
```

Install them manually:

```bash
python -m pip install psutil wandb
```
