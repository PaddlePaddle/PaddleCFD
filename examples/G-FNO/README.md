# G-FNO

## 1. Background

G-FNO (Group Equivariant Fourier Neural Operators) is a family of operator-learning models for PDE surrogate modeling. This PaddleCFD integration keeps the Paddle version of the main 2D/3D FNO, GCNN, GFNO, Ghybrid, and radialNO variants.

![G-FNO network](assets/network_visual.png)

## 2. Code Layout

- Core model code: `ppcfd/models/g_fno`
- Training and data generation scripts: `examples/G-FNO`

## 3. Installation

At the PaddleCFD repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

No extra Python packages beyond PaddleCFD root requirements are needed for the Paddle G-FNO runtime.

## 4. Import Models From Installed PaddleCFD

```python
from ppcfd.models.g_fno import FNO2d, GFNO2d

fno = FNO2d(
    num_channels=1,
    modes1=12,
    modes2=12,
    width=20,
    initial_step=10,
    grid_type="symmetric",
)

gfno = GFNO2d(
    num_channels=1,
    modes=12,
    width=10,
    initial_step=10,
    reflection=False,
    grid_type="symmetric",
)
```

## 5. Data Preparation

### 5.1 Navier-Stokes with Symmetric Forcing

From `examples/G-FNO/data_generation/navier_stokes`:

```bash
python "ns_2d_rt.py" --nu=1e-4 --T=30 --N=1200 --save_path="./data" --ntest=100 --period=4 --device=auto
```

### 5.2 Other datasets

- NS / PDEArena / PDEBench shallow-water datasets still follow the original upstream data sources referenced by the original paper.
- Place datasets under a local `data/` directory and pass absolute or repository-relative paths to `experiments.py`.

## 6. Training From Examples

From `examples/G-FNO`:

### NS

```bash
python "experiments.py" --seed=1 --data_path="./data/ns_V1e-4_N10000_T30.mat" \
    --results_path="./results/ns_V1e-4_N10000_T30.mat/GFNO2d_p4" --strategy=teacher_forcing \
    --T=20 --ntrain=1000 --nvalid=100 --ntest=100 --model_type=GFNO2d_p4 --modes=12 --width=10 \
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix="ns_V1e-4_N10000_T30.mat_GFNO2d_p4_seed1" \
    --learning_rate=1e-3 --early_stopping=100 --verbose --super \
    --super_path="./data/ns_data_V1e-4_N20_T50_R256test.mat" --device=auto
```

### NS-Sym

```bash
python "experiments.py" --seed=1 --data_path="./data/ns_V0.0001_N1200_T30_cos4.mat" \
    --results_path="./results/ns_V0.0001_N1200_T30_cos4.mat/GFNO2d_p4" --strategy=teacher_forcing \
    --T=10 --ntrain=1000 --nvalid=100 --ntest=100 --model_type=GFNO2d_p4 --modes=12 --width=10 \
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix="ns_V0.0001_N1200_T30_cos4.mat_GFNO2d_p4_seed1" \
    --learning_rate=1e-3 --early_stopping=100 --verbose --super \
    --super_path="./data/ns_V0.0001_N1200_T30_cos4_super.mat" --device=auto
```

### SWE (PDEArena)

```bash
python "experiments.py" --seed=1 --data_path="./data/ShallowWater2D" \
    --results_path="./results/ShallowWater2D/GFNO2d_p4" --strategy=teacher_forcing \
    --T=9 --ntrain=5600 --nvalid=1120 --ntest=1120 --model_type=GFNO2d_p4 --modes=32 --width=10 \
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix="ShallowWater2D_GFNO2d_p4_seed1" \
    --learning_rate=1e-3 --early_stopping=100 --verbose --time_pad --device=auto
```

### SWE-Sym (PDEBench)

```bash
python "experiments.py" --seed=1 --data_path="./data/2D_rdb_NA_NA.h5" \
    --results_path="./results/2D_rdb_NA_NA.h5/GFNO2d_p4" --strategy=teacher_forcing \
    --T=24 --ntrain=800 --nvalid=100 --ntest=100 --model_type=GFNO2d_p4 --modes=12 --width=10 \
    --batch_size=20 --epochs=100 --suffix=seed1 --txt_suffix="2D_rdb_NA_NA.h5_GFNO2d_p4_seed1" \
    --learning_rate=1e-3 --early_stopping=100 --verbose --super --device=auto
```

## 7. Notes

- `setup.sh` is intentionally not included in PaddleCFD.
- `run_experiment.sh` is intentionally not included in PaddleCFD.
- Removed Paddle-only unsupported model types that depend on Torch-only group-equivariant libraries remain unsupported here as well.
