# PROSE-FD

[PROSE-FD: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics](https://arxiv.org/abs/2409.09811). Accepted by 2024 NeurIPS Foundation Models for Science Workshop.

Pretrained PROSE-FD model weights can be found on https://huggingface.co/felix-lyx/prose.

## Import the model

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

## Quick Start

### Train

Single device, dryrun:

```bash
cd examples/prose_fd
python main.py dryrun=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd device=gpu:0 batch_size=2 batch_size_eval=1 num_workers=0 num_workers_eval=0 log_eval_plots=-1 exp_name=sw64_single_gpu exp_id=prose_fd_sw64_dryrun data.shallow_water.data_path="/path/to/your/data.h5"
```

To launch a model training with modified arguments (arg1,val1), (arg2,val2):

```bash
python main.py arg1=val1 arg2=val2
```

For example:

```bash
python "main.py" \
    use_wandb=0 \
    data=shallow_water_minimal \
    model=prose_2to1 \
    optim=wsd \
    device=gpu:0 \
    max_epoch=5 \
    n_steps_per_epoch=800 \
    batch_size=4 \
    batch_size_eval=6 \
    num_workers=0 \
    num_workers_eval=0 \
    log_eval_plots=-1 \
    exp_name=sw64_single_gpu \
    exp_id=prose_fd_sw64_ep5
```

All default arguments can be found in the `configs` folder, managed using [Hydra](https://hydra.cc/).

Scripts for reproducing the results in the paper are located in the `scripts` folder.

## Data

The dataset we used are collected from [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/pdearena/pdearena), and [CFDBench](https://github.com/luo-yining/CFDBench). More details about data preprocessing are included in `data_utils/README.md`.

## Convert pretrained weights

PyTorch pretrained weights can be converted to PaddlePaddle format:

```bash
cd examples/prose_fd
python tools/convert_torch_ckpt_to_paddle.py --torch-ckpt /path/to/model.pth --paddle-ckpt /path/to/model.pdparams
```

> Note: `tools/forward_pretrained_paddle.py` is not included in this integration. If needed, please refer to the original repository.

## Extra dependencies for PROSE-FD

- `wandb`: required only if `use_wandb=1`
- `torch`: required only for `tools/convert_torch_ckpt_to_paddle.py`

## Directory Structure

```
examples/prose_fd/
├── main.py                         # Training entry point
├── trainer.py                      # Trainer
├── evaluate.py                     # Evaluator
├── dataset.py                      # Dataset registry
├── configs/                        # Hydra YAML configs
│   ├── main.yaml                   # Main config
│   ├── data/                       # Data configs
│   ├── model/                      # Model configs
│   ├── optim/                      # Optimizer configs
│   └── symbol/                     # Symbol configs
├── data_utils/                     # Data loading & preprocessing
├── tools/
│   └── convert_torch_ckpt_to_paddle.py  # Weight conversion
├── utils/                          # Training utilities
└── README.md

ppcfd/models/prose_fd/
├── __init__.py                     # Public API exports
├── build_model.py                  # Model factory
├── transformer.py                  # Transformer encoder/decoder
├── transformer_wrappers.py         # PROSE_1to1 / PROSE_2to1
├── attention_utils.py              # Attention layers
├── embedder.py                     # Input embedders
├── paddle_utils.py                 # PaddlePaddle utilities
├── rotary_embedding_paddle.py      # Rotary position embeddings
├── runtime.py                      # Device helpers
└── symbol_utils/                   # Symbolic equation encoding
    ├── environment.py
    ├── encoders.py
    ├── generators.py
    └── node_utils.py
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
