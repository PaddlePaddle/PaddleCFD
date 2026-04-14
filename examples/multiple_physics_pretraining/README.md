# Multiple Physics Pretraining (MPP)

This example integrates the [MPP](https://openreview.net/forum?id=DKSI3bULiZ) (Multiple Physics Pretraining) model into PaddleCFD.

Multiple Physics Pretraining is a pretraining strategy in which multiple sets of dynamics are jointly normalized and embedded into a single space for prediction. It uses an **AViT** (Axial Vision Transformer) architecture that learns multiple physics simultaneously through pretraining, enabling strong finetuning performance even across different physics domains.

Paper: "Multiple Physics Pretraining for Spatiotemporal Surrogate Models" (NeurIPS 2024)

Below are quick instructions on paddle, full readme please visit https://github.com/PolymathicAI/multiple_physics_pretraining

## Installation

```bash
pip install ppcfd
pip install wandb  # optional
```

## Quick Start

### Import the model

```python
from ppcfd.models.multiple_physics_pretraining import AViT, build_avit
```

### Train (single device)

```bash
python train_basic.py --run_name my_experiment --config basic_config --yaml_config config/mpp_avit_ti_config.yaml
```

### Finetune from pretrained weights

Original PyTorch pretrained weights are available at:
https://drive.google.com/drive/folders/1Qaqa-RnzUDOO8-Gi4zlf4BE53SfWqDwx

To use them in PaddleCFD, first convert to PaddlePaddle format:

```bash
python convert_torch_weights.py \
    --yaml_config config/mpp_avit_s_config.yaml \
    --config basic_config \
    --weights path/to/MPP_AViT_S.tar \
    --output models_paddle/MPP_AViT_S.pdparams
```

Then finetune:

```bash
python train_basic.py --run_name my_finetune --config finetune --yaml_config config/mpp_avit_s_config.yaml
```

### Inference

If needed, use follow code to generate test input.

```bash
python multiple_physics_pretraining/generate_forward_case.py --output /tmp/case.npz --labels 0,1,2 --bcs 0,0 --output ./forward_case.npz
```

Then run a test forward case:

```bash
python forward_pretrained.py \
    --yaml_config config/mpp_avit_s_config.yaml \
    --config basic_config \
    --weights path/to/checkpoint.pdparams \
    --case_npz path/to/input.npz \
    --output path/to/output.npz
```

## Model Variants

| Variant   | embed_dim | num_heads | processor_blocks |
| --------- | --------- | --------- | ---------------- |
| Ti (Tiny) | 192       | 3         | 12               |
| S (Small) | 384       | 6         | 12               |
| B (Base)  | 768       | 12        | 12               |
| L (Large) | 1024      | 16        | 24               |

Config files are provided in `config/` for each variant. Use the `basic_config` namespace for pretraining and `finetune` for finetuning.

## Directory Structure

```
examples/multiple_physics_pretraining/
├── config/                      # YAML configuration files (Ti/S/B/L)
├── train_basic.py               # Training script
├── forward_pretrained.py        # Inference script
├── convert_torch_weights.py     # PyTorch -> PaddlePaddle weight conversion
├── requirements.txt             # Additional dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file

ppcfd/models/multiple_physics_pretraining/
├── avit.py                      # AViT model definition
├── shared_modules.py            # MLP, Attention, PositionBias
├── spatial_modules.py           # AxialAttention, hMLP stem/output
├── time_modules.py              # Temporal attention block
├── mixed_modules.py             # SpaceTimeBlock combiner
├── DropPath_util.py             # Stochastic depth
├── paddle_utils.py              # PaddlePaddle utilities
├── utils/                       # Training utilities
│   ├── YParams.py               # YAML config parser
│   ├── logging_utils.py         # Logging
│   ├── schedulers.py            # LR scheduler
│   ├── adan_paddle.py           # Adan optimizer
│   ├── dadapt_adam_paddle.py    # DAdaptAdam optimizer
│   ├── dadapt_adan_paddle.py    # DAdaptAdan optimizer
│   └── custom_optimizer_base.py # Optimizer base class
└── data_utils/                  # Data loading
    ├── datasets.py              # MixedDataset, dataset registry
    ├── hdf5_datasets.py         # HDF5 dataset classes (SWE, NS, etc.)
    └── mixed_dset_sampler.py    # Multi-dataset sampler
```

## Adding Datasets

Datasets must return data in `(Batch, Time, Channel, H, W)` format and extend `BaseHDF5DirectoryDataset`. See `data_utils/hdf5_datasets.py` for examples.

1. Define your dataset class in `ppcfd/models/multiple_physics_pretraining/data_utils/hdf5_datasets.py`
2. Register it in `DSET_NAME_TO_OBJECT` in `datasets.py`
3. Add data paths to the config YAML file

## Citing

```bibtex
@inproceedings{
  mccabe2024multiple,
  title={Multiple Physics Pretraining for Spatiotemporal Surrogate Models},
  author={Michael McCabe and Bruno R{\'e}galdo-Saint Blancard and others},
  booktitle={NeurIPS},
  year={2024},
  url={https://openreview.net/forum?id=DKSI3bULiZ}
}
```
