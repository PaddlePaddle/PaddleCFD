# NOMAD: Nonlinear Manifold Decoders for Operator Learning

## 1. Introduction

NOMAD learns nonlinear low-dimensional solution manifolds for operator-learning tasks. This PaddleCFD example migrates the original PyTorch NOMAD examples to PaddlePaddle and includes three cases:

- `antiderivative`: learn the antiderivative operator on synthetic one-dimensional functions.
- `advection`: learn the one-dimensional pure advection solution operator.
- `shallowwater`: learn the shallow water equation solution operator.

The model code is under `ppcfd/models/nomad`, while data loading, training, inference, and evaluation scripts are under `examples/nomad`.

## 2. Directory

```text
PaddleCFD/
|-- ppcfd/models/nomad/
|   |-- antiderivative/nomad_antiderivative.py
|   |-- advection/mlp.py
|   |-- advection/operator_model.py
|   `-- shallowwater/nomad_model.py
`-- examples/nomad/
    |-- antiderivative/
    |   |-- data_utils.py
    |   |-- train.py
    |   |-- inference.py
    |   `-- evaluate.py
    |-- advection/
    |   |-- dataset.py
    |   |-- train.py
    |   |-- inference.py
    |   |-- evaluate.py
    |   `-- pure_advection_traintest.npz
    `-- shallowwater/
        |-- dataset.py
        |-- train.py
        |-- inference.py
        |-- evaluate.py
        |-- train_SW.npz
        `-- test_SW.npz
```

## 3. Environment

Install PaddleCFD and dependencies from the repository root:

```bash
cd PaddleCFD
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use a GPU-enabled PaddlePaddle package for full training and performance evaluation. CPU can be used for syntax checks and small debugging runs, but the default training iterations are set to reproduce the original experiment scale.

## 4. Data and Pretrained Checkpoints

The dataset and checkpoint files are hosted on BOS and are not tracked in the repository. Download them before running training, inference, or evaluation, and place each file at the path shown below.

### Datasets

The antiderivative case generates training and test samples on the fly, so no extra dataset is required.

The advection case requires `examples/nomad/advection/pure_advection_traintest.npz`:

```bash
cd examples/nomad/advection
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/nomad/pure_advection_traintest.npz
```

The shallow water case requires `train_SW.npz` and `test_SW.npz` under `examples/nomad/shallowwater/`:

```bash
cd examples/nomad/shallowwater
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/nomad/train_SW.npz
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/nomad/test_SW.npz
```

### Pretrained Checkpoints

To run inference and evaluation without training from scratch, download the pretrained checkpoints into the `checkpoints/` directory of each case:

```bash
# antiderivative
mkdir -p examples/nomad/antiderivative/checkpoints
wget -P examples/nomad/antiderivative/checkpoints \
  https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/nomad/nomad_antiderivative.pdparams

# advection
mkdir -p examples/nomad/advection/checkpoints
wget -P examples/nomad/advection/checkpoints \
  https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/nomad/nomad_advection_model.pdparams

# shallow water
mkdir -p examples/nomad/shallowwater/checkpoints
wget -P examples/nomad/shallowwater/checkpoints \
  https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/nomad/nomad_sw_nonlinear_n10.pdparams
```

Download links summary:

- Datasets
    - `pure_advection_traintest.npz`: https://paddle-org.bj.bcebos.com/paddlecfd/datasets/nomad/pure_advection_traintest.npz
    - `train_SW.npz`: https://paddle-org.bj.bcebos.com/paddlecfd/datasets/nomad/train_SW.npz
    - `test_SW.npz`: https://paddle-org.bj.bcebos.com/paddlecfd/datasets/nomad/test_SW.npz
- Checkpoints
    - `nomad_advection_model.pdparams`: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/nomad/nomad_advection_model.pdparams
    - `nomad_antiderivative.pdparams`: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/nomad/nomad_antiderivative.pdparams
    - `nomad_sw_nonlinear_n10.pdparams`: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/nomad/nomad_sw_nonlinear_n10.pdparams

## 5. Training

All commands below are run from the corresponding example directory. The first argument is the latent dimension `n`; the second argument is the decoder type, either `linear` or `nonlinear`.

### Antiderivative

```bash
cd examples/nomad/antiderivative
python train.py 10 nonlinear
```

The checkpoint is saved to:

```text
checkpoints/nomad_antiderivative.pdparams
```

For a quick environment smoke test, reduce the training iterations:

```bash
python train.py 10 nonlinear --iterations 2
```

### Advection

```bash
cd examples/nomad/advection
python train.py 10 nonlinear
```

The checkpoint is saved to:

```text
checkpoints/nomad_advection_model.pdparams
```

For a quick environment smoke test:

```bash
python train.py 10 nonlinear --iterations 2 --batch-size 4
```

For a reproducible run with full training-state checkpoints:

```bash
python train.py 10 nonlinear \
  --seed 0 \
  --train-seed 1234 \
  --test-seed 1234 \
  --state-path checkpoints/nomad_advection_train_state.pdopt \
  --state-interval 5000
```

To resume the complete training state, including optimizer, scheduler, data
sampler RNGs, and global step:

```bash
python train.py 10 nonlinear \
  --resume-state \
  --state-path checkpoints/nomad_advection_train_state.pdopt
```

### Shallow Water

```bash
cd examples/nomad/shallowwater
python train.py 10 nonlinear
```

The checkpoint is saved to:

```text
checkpoints/nomad_sw_nonlinear_n10.pdparams
```

For a quick environment smoke test:

```bash
python train.py 10 nonlinear --iterations 2
```

To save a candidate checkpoint without replacing the default one:

```bash
python train.py 10 nonlinear \
  --iterations 22060 \
  --checkpoint /tmp/nomad_sw_candidate.pdparams \
  --state-path /tmp/nomad_sw_candidate.pdopt \
  --state-interval 10000 \
  --seed 0 \
  --train-seed 1234 \
  --test-seed 1234
```

To resume the complete Shallow Water training state:

```bash
python train.py 10 nonlinear \
  --resume-state \
  --state-path /tmp/nomad_sw_candidate.pdopt
```

## 6. Inference

### Antiderivative

```bash
cd examples/nomad/antiderivative
python inference.py 10 nonlinear
```

The prediction is saved to:

```text
results/pred.npy
```

### Advection

```bash
cd examples/nomad/advection
python inference.py 10 nonlinear
```

The prediction is saved to:

```text
results/prediction.npz
```

To reduce memory usage, change the inference batch size:

```bash
python inference.py 10 nonlinear --num-test 1000 --infer-batch-size 50
```

To evaluate a candidate checkpoint without replacing the default one:

```bash
python inference.py 10 nonlinear \
  --checkpoint /path/to/nomad_advection_candidate.pdparams \
  --num-test 1000 \
  --infer-batch-size 100
```

### Shallow Water

```bash
cd examples/nomad/shallowwater
python inference.py 10 nonlinear
```

The prediction is saved to:

```text
results/pred_test.npy
```

## 7. Evaluation

Run evaluation after inference.

### Antiderivative

```bash
cd examples/nomad/antiderivative
python evaluate.py 10 nonlinear
```

The script prints the mean, standard deviation, minimum, and maximum relative L2 error.

### Advection

```bash
cd examples/nomad/advection
python evaluate.py
```

The script prints the mean, standard deviation, minimum, and maximum relative L2 error.

### Shallow Water

```bash
cd examples/nomad/shallowwater
python evaluate.py
```

The script prints the mean relative L2 error for `rho`, `u`, and `v`.

## 8. Accuracy Alignment Checklist

Use this table to record acceptance results before submitting the PaddleCFD pull request.

| Case | Decoder | `n` | Forward loss diff | 2+ epoch loss trend | Metric diff | Status |
| --- | --- | ---: | ---: | --- | ---: | --- |
| Antiderivative | nonlinear | 10 | `2.38e-07` | aligned in 2-step check | `0.000565` abs mean diff | pass |
| Advection | nonlinear | 10 | `1.19e-07` | aligned in 2-step check | `0.003383` abs mean diff | improving |
| Shallow Water | nonlinear | 10 | `5.96e-08` | aligned in 2-step check | `rho/u/v <= 0.000796` abs mean diff | pass |

Latest Advection candidate note: a fixed-seed full run with `--seed 0 --train-seed 1234 --test-seed 1234` produced `0.015975` mean error. The `0.021176` comparison target comes from the author-published Google Drive `Error_Vectors` package referenced by the official NOMAD README.

Latest Shallow Water note: the default checkpoint uses the 22,060-step candidate from the corrected scheduler/test-sampler training script. Its mean errors are `rho=0.000875`, `u=0.034652`, and `v=0.038670`, all within the configured absolute `0.002` tolerance against the Google Drive reference vectors.

### Reproducibility note

The original NOMAD training scripts initialize model weights from `np.random.randint(...)` and sample training batches with a JAX PRNG stream. The author-published `Error_Vectors` therefore represent particular stochastic training runs rather than a seed-stable deterministic target. This Paddle migration verifies the implementation itself by assigning identical JAX/stax weights and inputs to the Paddle model; the forward loss, prediction, gradient, and two-step optimization traces align at about `1e-7` level across Antiderivative, Advection, and Shallow Water.

Consequently, small differences in final training metrics can be caused by unmatched random initialization, batch order, and framework-level optimizer numerics even when the migrated model is correct. For strict metric reproduction, the reference run seed, initialization seeds, sampler state, optimizer state, and checkpoint-selection rule must be fixed and shared. Without those, a `0.2%` relative tolerance against one released error vector is not a deterministic property of the model implementation.

Recommended acceptance targets:

- Single-card forward loss difference: `1e-4` level.
- Backward alignment: train at least 2 epochs or equivalent repeated optimization steps and compare loss trends.
- Dataset metric alignment: metric difference within `0.2%`.
- Compiler performance: compare Paddle deep learning compiler on/off and record average speedup.

## 9. Compiler Performance

Benchmark training or inference twice with the same checkpoint, batch size, and device. Keep the first several iterations as warmup and report average step time after warmup.

Recommended command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \
python examples/nomad/benchmark_compiler.py \
  --case antiderivative \
  --decoder nonlinear \
  --n 10 \
  --batch-size 1024 \
  --points 500 \
  --warmup 30 \
  --steps 200 \
  --repeats 3 \
  --output examples/nomad/compiler_benchmark_report.json
```

Current record:

| Case | Mode | Batch | Points | Warmup steps | Measured steps | Repeats | Dynamic avg step | CINN avg step | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Antiderivative | inference | 1024 | 500 | 30 | 200 | 3 | `0.009102s` | `0.006728s` | `35.29%` |

## 10. Reference

- Seidman, J. H., Kissas, G., Perdikaris, P., & Pappas, G. J. NOMAD: Nonlinear Manifold Decoders for Operator Learning. arXiv:2206.03551, 2022.
- Original code repository: https://github.com/PredictiveIntelligenceLab/NOMAD
