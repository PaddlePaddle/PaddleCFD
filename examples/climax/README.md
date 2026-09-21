# ClimaX — A Weather and Climate Foundation Model

This example integrates [ClimaX](https://arxiv.org/abs/2301.10343), a foundation model for weather and climate, into PaddleCFD. The model is reimplemented in PaddlePaddle and **only** set up for its **climate-projection downstream task**: on [ClimateBench](https://doi.org/10.5281/zenodo.5196512), ClimaX emulates the surface air temperature (`tas`) response to greenhouse-gas and aerosol forcings (`CO2`, `SO2`, `CH4`, `BC`) on a 5.625 degree grid.

## Model architecture

ClimaX applies a Vision Transformer to an arbitrary set of input variables:

| Stage                        | Description                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Per-variable patch embedding | Each input variable goes through its own `PatchEmbed` (or all variables share weights via `parallel_patch_embed`)                        |
| Variable aggregation         | Per-variable token sequences are pooled into a single sequence by cross-attention against a learned query                                |
| ViT backbone                 | Standard transformer encoder blocks (`embed_dim` / `depth` / `num_heads`) with 2D sincos positional embedding and lead-time conditioning |
| Head                         | A linear head maps the encoded tokens back to the output grid                                                                            |

The climate-projection variant `ClimaXClimateBench` adds a temporal stage: a time positional embedding and a cross-attention module pool the input history window (`time_history` steps) before the linear head.

```python
from ppcfd.models.climax import ClimaX, ClimaXClimateBench

model = ClimaXClimateBench(
    default_vars=["CO2", "SO2", "CH4", "BC"],
    out_vars="tas",
    img_size=[32, 64],  # 5.625 degree grid
    time_history=10,
    patch_size=2,
    embed_dim=1024,
    depth=8,
    num_heads=16,
)
```

## Pretrained checkpoint

The original released pretrained ClimaX weights are hosted on HuggingFace at <https://huggingface.co/tungnd/climax>; the [5.625deg checkpoint](https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt) matches the resolution used in this example.

The paddle training script loads PaddlePaddle `.pdparams` files; a torch `.ckpt` checkpoint must be converted to `.pdparams` before use.

## Quick start

### Data preparation

Download the ClimateBench dataset from Zenodo ([DOI 10.5281/zenodo.5196512](https://doi.org/10.5281/zenodo.5196512), version 1.0): fetch `train_val.tar.gz` and `test.tar.gz` and extract both under `data/climatebench/`. Following the original ClimateBench experiment, regrid the data to 5.625 degree with the bundled script:

```bash
# run from examples/climax
python regrid_climatebench.py data/climatebench/train_val \
    --save_path data/climatebench/5.625deg/train_val --ddeg_out 5.625
python regrid_climatebench.py data/climatebench/test \
    --save_path data/climatebench/5.625deg/test --ddeg_out 5.625
```

The expected directory layout (relative to `examples/climax`, matching the default `data.root_dir`):

```text
data/climatebench/5.625deg
├── train_val
│   ├── inputs_historical.nc
│   ├── inputs_ssp126.nc
│   ├── inputs_ssp370.nc
│   ├── inputs_ssp585.nc
│   ├── inputs_hist-GHG.nc
│   ├── inputs_hist-aer.nc
│   └── outputs_*.nc   # one per simulation above
└── test
    ├── inputs_ssp245.nc
    └── outputs_ssp245.nc
```

### Training

All knobs live in `configs/climate_projection.yaml`; there is no CLI surface:

```bash
cd examples/climax
python train.py
```

Edit at least these fields first:

| Field                   | Meaning                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `data.root_dir`         | ClimateBench 5.625 degree dataset root (contains `train_val/` and `test/`)                                 |
| `model.pretrained_path` | Released checkpoint (`.pdparams`) to finetune from; leave empty to train from scratch                      |
| `model.init_state_path` | Optional full initial state (backbone + heads) in `.pdparams` form; when set, `pretrained_path` is ignored |

Checkpoints (`best.pdparams` / `last.pdparams`) are written to `train.default_root_dir`. Training monitors `val/w_mse`, keeps the best checkpoint, and stops early after `train.patience` epochs without improvement.

### Evaluation

Evaluation is built into `train.py`: when training finishes, the best checkpoint is restored automatically and evaluated on the test split, printing latitude-weighted metrics (w-mse, w-rmse and normalized w-rmse for `tas`).

## CINN acceleration

A single environment variable engages the CINN compiler, with no code change:

```bash
cd examples/climax
CLIMAX_USE_CINN=1 python train.py
```

`CLIMAX_USE_CINN=1` sets the CINN-related FLAGS (`FLAGS_prim_enable_dynamic`, `FLAGS_prim_all`, `FLAGS_use_cinn`) before `import paddle` and wraps the model with `paddle.jit.to_static(full_graph=True)`. With the switch off (default), training runs the pure dynamic-graph path.

Measured on this example, steady-state training is **5.2% faster** under CINN (RTX 4060 Ti, FP32, batch_size=1, 793 steps per epoch). Compilation is a one-off cost of about 90 s on the same setup. These numbers are measured on this small-scale climate-projection configuration.

## Accuracy alignment

- Forward pass: against the torch reference under the same weights on a fixed input, the paddle output matches with a mean absolute difference of 2.24e-08.
- Training: a 5-epoch run from the same initial weights (same seed, FP32) reaches final test metrics within 5.1% of the torch baseline, and the loss difference between the two frameworks fluctuates around 0, with consistent trends.

## Additional dependencies

Beyond core PaddleCFD, this example needs the xarray / netCDF data stack:

```bash
python -m pip install xarray netCDF4 scipy
```

The regrid script additionally uses `xesmf`, which is installed through conda-forge (`conda install -c conda-forge xesmf`).

## Citing

If you find ClimaX useful in your research, please consider citing:

```bibtex
@article{nguyen2023climax,
  title={ClimaX: A foundation model for weather and climate},
  author={Nguyen, Tung and Brandstetter, Johannes and Kapoor, Ashish and Gupta, Jayesh K and Grover, Aditya},
  journal={arXiv preprint arXiv:2301.10343},
  year={2023}
}
```
