# PROSE-FD

PROSE-FD is a multimodal PDE foundation model that learns multiple operators for forecasting fluid dynamics. It combines spatio-temporal field data (velocity, pressure, density, etc.) with symbolic equation representations (encoded as token sequences) in a unified Transformer, supporting both data-only (`prose_1to1`) and data+symbol (`prose_2to1`) inference paradigms.

## Model Architecture

PROSE-FD (`PROSE_2to1`) is a multimodal Transformer with the following pipeline:

- **Embedder** — patchifies spatial fields and projects them into the embedding dimension (linear or conv patchification), fused with learnable time embeddings and patch position embeddings.
- **Data Encoder** — a Transformer encoder over the embedded data sequence.
- **Symbol Encoder** — a Transformer encoder over the encoded symbolic equation token sequence.
- **Fusion** — cross-modal fusion of the data and symbol representations.
- **Operator Decoder** — an operator-style decoder (cross-attention from time-space queries to the fused representation) that produces the predicted future fields.

The `prose_1to1` variant drops the symbol branch and conditions on data only. The whole model is configured via Hydra (`configs/`), and the public API is exposed through the `ppcfd.models.prose_fd` package.

### Import the model

```python
from omegaconf import OmegaConf
from ppcfd.models.prose_fd import PROSE_2to1, SymbolicEnvironment

model_cfg = OmegaConf.load("examples/prose_fd/configs/model/prose_2to1.yaml")
data_cfg = OmegaConf.load("examples/prose_fd/configs/data/shallow_water_minimal.yaml")
symbol_cfg = OmegaConf.load("examples/prose_fd/configs/symbol/symbol.yaml")

symbol_env = SymbolicEnvironment(symbol_cfg)
model = PROSE_2to1(
    model_cfg,
    symbol_env,
    data_cfg.x_num,
    data_cfg.max_output_dimension,
    data_cfg.t_num - 10,
)
```

## Pretrained Model

Pretrained PROSE-FD (PaddlePaddle) weights are available on AI Studio:

https://aistudio.baidu.com/modelsdetail/49246?modelId=49246

Load them in training/evaluation with `reload_model=/path/to/prose_fd.pdparams` (or `eval_from_exp=<dir>` for evaluation only).

## Installation

PROSE-FD is part of PaddleCFD. From the repository root:

```bash
pip install -e .                # installs the ppcfd package
pip install -r requirements.txt
```

## Data

PROSE-FD trains on shallow-water (and broader fluids) datasets collected from [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/pdearena/pdearena) and [CFDBench](https://github.com/luo-yining/CFDBench). Two shallow-water configurations are provided and tested:

| Config | Spatial resolution | Use |
|---|---|---|
| `data=shallow_water_minimal` | 64×64 | fast test / single-dataset debugging |
| `data=fluids` (+ `data.shallow_water.x_num=128`) | 128×128 (PDEBench full) | full multi-operator training |

Point the loader at your data with `data.shallow_water.data_path=/path/to/2D_rdb_NA_NA.h5`. Preprocessing details live in `data_utils/`.

## Quick Start

Run from `examples/prose_fd/`.

### Test (single GPU, dryrun)

```bash
python main.py dryrun=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd \
  device=gpu:0 batch_size=2 batch_size_eval=1 num_workers=0 num_workers_eval=0 \
  log_eval_plots=-1 exp_name=sw64_smoke data.shallow_water.data_path=/path/to/data.h5
```

### Full training

```bash
python main.py use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd device=gpu:0 \
  max_epoch=5 n_steps_per_epoch=800 batch_size=4 num_workers=0 log_eval_plots=-1 \
  exp_name=sw64_train data.shallow_water.data_path=/path/to/data.h5
```

Any argument can be overridden with `key=value`; defaults live in `configs/`.

### Inference / evaluation only

Pure inference (no training loop) is entered with `eval_only=1`. The path to load is given by `eval_from_exp`: if `<eval_from_exp>/checkpoint.pth` exists it is used, otherwise `eval_from_exp` is treated as a weight file directly. So you can point it at either an experiment dump directory or a downloaded `.pdparams` file (e.g. the AI Studio weights linked above).

Standard operator-mode evaluation on the shallow-water config:

```bash
python main.py eval_only=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd \
  device=gpu:0 batch_size_eval=1 num_workers_eval=0 log_eval_plots=-1 \
  exp_name=sw64_eval eval_from_exp=/path/to/prose_fd.pdparams \
  data.shallow_water.data_path=/path/to/data.h5
```

Rollout-in-time evaluation (autoregressive step-by-step extrapolation instead of one-shot operator prediction) — add `rollout=1`:

```bash
python main.py eval_only=1 rollout=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd \
  device=gpu:0 batch_size_eval=1 num_workers_eval=0 log_eval_plots=-1 \
  exp_name=sw64_rollout eval_from_exp=/path/to/prose_fd.pdparams \
  data.shallow_water.data_path=/path/to/data.h5
```

Reported metrics are controlled by `validation_metrics_print` (rel L2, per-step L2, etc.); set `print_outputs=1` to dump predicted-vs-ground-truth figures under `eval_dump_path`.

### CINN accelerated training

The PaddlePaddle native compiler CINN can accelerate training once the model is converted to a static graph (`paddle.jit.to_static`). This example uses a single environment variable as the switch — it injects the CINN-related FLAGS before `import paddle` and wraps the assembled model with `to_static(full_graph=True)`:

```bash
PROSE_TO_STATIC=1 python main.py use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd \
  device=gpu:0 batch_size=2 num_workers=0 log_eval_plots=-1 \
  exp_name=sw64_cinn data.shallow_water.data_path=/path/to/data.h5
```

CINN is active once the log shows `Compiling subgraph with CINN backend`.

Measured speedup (RTX 4060 Ti, batch=2, fp32, shallow-water 64×64, steady-state over 1600 steps):

| Mode | steady-state pure-train step/s |
|---|---|
| dynamic graph (baseline) | 1.98 |
| CINN to_static | 2.11 |

**+6.3% steady-state pure-training speedup.** CINN compiles the graph once (~100s warmup); short runs may look slower end-to-end, but steady-state and long training yield a net win.

## Directory Structure

```
examples/prose_fd/
├── main.py              # training entry point (CINN FLAGS injected at top)
├── trainer.py           # trainer (optional PROSE_BENCHMARK pure-train timing)
├── evaluate.py          # evaluator
├── dataset.py           # dataset registry
├── configs/             # Hydra YAML configs (main / data / model / optim / symbol)
├── data_utils/          # data loading & preprocessing
└── utils/               # training utilities

ppcfd/models/prose_fd/
├── build_model.py            # model factory (to_static wrapping)
├── transformer_wrappers.py   # PROSE_1to1 / PROSE_2to1
├── transformer.py            # transformer encoder / decoder / fusion
├── attention_utils.py        # attention layers
├── embedder.py               # input embedders (PatchTokensToGrid)
├── runtime.py                # device helpers
└── symbol_utils/             # symbolic equation encoding
```

## Citation

```bibtex
@article{liu2024prose_fd,
  title={{PROSE-FD}: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics},
  author={Liu, Yuxuan and Sun, Jingmin and He, Xinjie and Pinney, Griffin and Zhang, Zecheng and Schaeffer, Hayden},
  journal={arXiv preprint arXiv:2409.09811},
  year={2024}
}
```
