# CoNFILD

## 1.Background

CoNFILD (Conditional Neural Field Latent Diffusion) is an AI-assisted framework for generating high-fidelity spatiotemporal turbulent flow fields. It combines Conditional Neural Fields (CNF) with Latent Diffusion Models to achieve efficient compression and probabilistic modeling of complex turbulent flows. The method supports zero-shot conditional generation tasks such as sparse sensor reconstruction and super-resolution.

### Key Features

- **Efficient Compression**: Neural fields compress high-dimensional flow fields into low-dimensional latent representations with compression ratios of 0.002%-0.017%
- **Probabilistic Modeling**: Latent diffusion process learns statistical distributions and dynamic characteristics of turbulence
- **Zero-Shot Inference**: Enables conditional generation without retraining through Bayesian posterior sampling
- **Modular Design**: Model code in `ppcfd/models/confild`, training/inference code in `examples/confild`

### Reference Paper

| Year | Journal | Authors | Citations | Link |
|------|---------|---------|-----------|------|
| 2024 | Nature Communications | Pan Du, Meet Hemant Parikh, Xiantao Fan, Xin-Yang Liu, Jian-Xun Wang | 15 | [Paper](https://doi.org/10.1038/s41467-024-54712-1) |

***

## 2.Model Description

CoNFILD employs a two-stage training framework:

### Stage 1: Conditional Neural Field (CNF) - SIREN with FiLM

**Mathematical Formulation**:
$$
\mathcal{E}(\mathbf{X},\mathbf{L}) = \text{SIREN}(\mathbf{x}) + \text{FILM}(\mathbf{L})
$$

- **SIREN Network**: Uses sinusoidal activation functions to capture periodic features
- **FiLM Modulation**: Conditions each layer's bias through latent vector $\mathbf{L}$
- **Functionality**: Compresses high-dimensional spatiotemporal flow fields into low-dimensional latent representations

### Stage 2: Latent Diffusion Model

**Forward Process**: Gradually adds Gaussian noise
$$
q(\mathbf{z}_t | \mathbf{z}_{t-1}) = \mathcal{N}(\mathbf{z}_t; \sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I})
$$

**Reverse Process**: Trains U-Net to predict noise
$$
\mathbf{z}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{z}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{z}_t, t) \right) + \sigma_t \epsilon
$$

### Zero-Shot Conditional Generation

Based on sparse observations $\Psi$, corrects latent space sampling through gradients:
$$
\nabla_{\mathbf{z}_t} \log p(\mathbf{z}_t|\Psi) \approx \nabla_{\mathbf{z}_t} \log p(\Psi|\mathbf{z}_t) + \nabla_{\mathbf{z}_t} \log p(\mathbf{z}_t)
$$

***

## 3.Dataset

### 3.1 Complete Dataset Download

[Download complete dataset from AI Studio](https://aistudio.baidu.com/dataset/detail/357298) - includes both CNF training data and extracted diffusion latent codes.

**Complete directory structure**:
```
data/
├── Case1/
│   ├── data.npy           # CNF: Flow field data [N_samples, spatial_dims..., features]
│   ├── coords.npy         # CNF: Query coordinates [spatial_dims..., coord_features]
│   ├── train_data.npy     # Diffusion: Training latent codes from CNF
│   ├── valid_data.npy     # Diffusion: Validation latent codes from CNF
│   ├── cnf/
│   │   └── normalizer_params.pdparams  # (Optional) Pre-trained normalizer
│   └── diffusion/
│       ├── ema.pdparams   # (Optional) Pre-trained diffusion model
│       └── ema.pkl
│
├── Case2/
│   ├── data.npy           # Channel flow data (no coords - uses dynamic grid)
│   ├── train_data.npy
│   ├── valid_data.npy
│   ├── cnf/
│   │   └── normalizer_params.pdparams
│   └── diffusion/
│       └── ema.pkl
│
├── Case3/
│   ├── data.npy
│   ├── coords.npy
│   ├── train_data.npy
│   ├── valid_data.npy
│   ├── cnf/
│   │   └── normalizer_params.pdparams
│   └── diffusion/
│       └── ema.pkl
│
└── Case4/
    ├── data.npy
    ├── coords.npy
    ├── train_data.npy
    ├── valid_data.npy
    ├── cnf/
    │   └── normalizer_params.pdparams
    └── diffusion/
        ├── ema.pdparams
        └── ema.pkl
```

**Notes**:
- **Case2** (channel flow) doesn't have `coords.npy` as it uses dynamically generated structured grid coordinates
- Pre-trained model checkpoints (`cnf/` and `diffusion/` subdirectories) are optional
- You can train from scratch using only `data.npy`, `coords.npy` (if applicable), `train_data.npy`, and `valid_data.npy`

### 3.2 Data File Descriptions

#### CNF Training Files
- `data.npy`: Original flow field snapshots for CNF to compress
- `coords.npy`: Spatial coordinates (not needed for Case2 which uses structured grids)

#### Diffusion Training Files
- `train_data.npy`: Extracted latent codes from trained CNF model (training set)
- `valid_data.npy`: Extracted latent codes from trained CNF model (validation set)

#### Optional Pre-trained Models
- `cnf/normalizer_params.pdparams`: Pre-trained data normalizer
- `diffusion/ema.pdparams` & `ema.pkl`: Pre-trained diffusion model with EMA

### 3.3 Data Split

- Training: 70%
- Validation: 15% (for model selection)
- Testing: 15% (for final evaluation)

Data is automatically shuffled before splitting to ensure randomness.

***

## 4.Model Training & Testing

### 4.1 Environment Setup

```bash
pip install scipy scikit-learn matplotlib hydra-core omegaconf
```

### 4.2 Configuration Parameters

#### CNF Model Configuration (confild_main.yaml)

```yaml
# Model Configuration
CONFILD:
  num_hidden_layers: 10          # Number of hidden layers
  out_features: 3                # Output features (e.g., u,v,p)
  hidden_features: 128           # Hidden layer features
  in_coord_features: 2           # Coordinate input features
  in_latent_features: 128        # Latent vector dimension

# Latent Container Configuration
Latent:
  N_samples: 16000               # Total number of samples
  N_features: 128                # Latent vector dimension
  dims: 2                        # Spatial dimension
  lumped: True                   # Dimension organization

# Training Configuration
TRAIN:
  batch_size: 64
  test_batch_size: 256
  epochs: 9800
  num_gpus: 1                    # Number of GPUs (>1 enables distributed training)
  num_workers: 4                 # Data loading workers
  lr:
    cnf: 1.e-4                   # CNF learning rate
    latents: 1.e-5               # Latent vector learning rate

# Data Configuration
DATA:
  data_path: data/Case1/data.npy      # Flow field data path
  coor_path: data/Case1/coords.npy    # Coordinate data path
  normalizer:
    method: "-11"                # Normalization method: "-11", "01", "ms", "none"
    dim: 0                       # Normalization dimension
```

#### Diffusion Model Configuration (un_confild_main.yaml)

```yaml
# UNet Model Configuration
UNET:
  image_size: 128                # Latent space spatial size
  in_channels: 128               # Input latent feature dimension
  out_channels: 128              # Output latent feature dimension
  num_channels: 128              # UNet base channels
  num_res_blocks: 2              # Number of residual blocks
  num_heads: 4                   # Number of attention heads
  num_head_channels: 64          # Channels per head
  attention_resolutions: "32,16,8"  # Attention resolutions

# Diffusion Process Configuration
Diff:
  steps: 1000                    # Diffusion steps
  noise_schedule: "cosine"       # Noise schedule strategy

# Training Configuration
TRAIN:
  batch_size: 16
  test_batch_size: 16
  ema_rate: "0.9999"             # EMA decay rate
  lr: 5.e-5                      # Learning rate
  lr_anneal_steps: 10000         # Learning rate annealing steps
  max_steps: 10000               # Maximum training steps
```

### 4.3 Multi-Case Support

CoNFILD provides pre-configured settings for 4 different flow cases:

| Case | Flow Type | Spatial Dim | Features | Samples | Hidden Layers | Latent Dim | Description |
|------|-----------|-------------|----------|---------|---------------|------------|-------------|
| **Case1** | Elbow flow | 2D | 3 (u,v,p) | 16000 | 10 | 128 | 2D elbow geometry flow |
| **Case2** | Channel flow | 2D | 4 (u,v,w,p) | 1200 | 10 | 256 | 3D channel flow (2D slice) |
| **Case3** | Periodic hill | 2D | 2 (u,v) | 2880 | 117 | 256 | Periodic hill flow |
| **Case4** | 3D flow | 3D | 3 (u,v,w) | 1200 | 15 | 384 | Full 3D turbulent flow |

**Select a case using the `--config-name` parameter**:

```bash
# Default: use confild_main.yaml (Case1)
python confild_main.py mode=train

# Use specific case configuration
python confild_main.py --config-name=confild_case1 mode=train  # Elbow flow
python confild_main.py --config-name=confild_case2 mode=train  # Channel flow
python confild_main.py --config-name=confild_case3 mode=train  # Periodic hill
python confild_main.py --config-name=confild_case4 mode=train  # 3D flow
```

### 4.4 Train CNF Model (Stage 1)

```bash
cd examples/confild

# Single GPU training (default Case1)
python confild_main.py mode=train

# Train specific case
python confild_main.py --config-name=confild_case2 mode=train

# Multi-GPU distributed training (4 GPUs)
python -m paddle.distributed.launch --gpus 0,1,2,3 confild_main.py mode=train TRAIN.num_gpus=4

# Multi-GPU for specific case
python -m paddle.distributed.launch --gpus 0,1,2,3 confild_main.py --config-name=confild_case3 mode=train TRAIN.num_gpus=4
```

**Notes on Distributed Training**:
- `TRAIN.num_gpus` must match the number of GPUs specified in `--gpus`
- Total batch size = batch_size × num_gpus
- Recommended 1-2 data loading workers per GPU

### 4.5 Test CNF Model

```bash
# Test with best model (default case)
python confild_main.py mode=test

# Test specific case
python confild_main.py --config-name=confild_case2 mode=test

# Test with specific checkpoint
python confild_main.py mode=test checkpoint=outputs/path/to/model
```

### 4.6 Train Diffusion Model (Stage 2)

```bash
# Train diffusion model (default Case1)
python un_confild_main.py mode=train

# Train for specific case
python un_confild_main.py --config-name=un_confild_case1 mode=train  # Case1: 128-dim latent
python un_confild_main.py --config-name=un_confild_case2 mode=train  # Case2: 256-dim latent
python un_confild_main.py --config-name=un_confild_case3 mode=train  # Case3: 256-dim latent
python un_confild_main.py --config-name=un_confild_case4 mode=train  # Case4: 384-dim latent
```

**Note**: Ensure the latent dimension matches the CNF model's latent dimension for the corresponding case.

### 4.7 Test Diffusion Model

```bash
# Generate new samples (default case)
python un_confild_main.py mode=test

# Test specific case
python un_confild_main.py --config-name=un_confild_case2 mode=test

# Use specific checkpoint
python un_confild_main.py mode=test checkpoint=outputs/path/to/unet_best.pdparams
```

***

## 5.Results

After training, the `outputs/` directory will contain:

**CNF Model Outputs**:
- `cnf_model_*.pdparams`: CNF model weights
- `latents_model_*.pdparams`: Latent vector weights
- `case.png`: Training loss curve
![](https://ai-studio-static-online.cdn.bcebos.com/1f81af1d579b4b41a525f867ac0fde19d59fb6fc44f8406aa84345c6015938c9)

**Diffusion Model Outputs**:
- `unet.pdparams`: U-Net model weights
- `loss_curve.png`: Training and validation loss curves
![](examples\confild\images\loss_curve.png)

**Evaluation Metrics**:
- Velocity field MSE: ~0.041
- Compression ratio: 0.002%-0.017%

**CNF Testing**:
- Test MSE and MAE metrics
- Per-sample prediction accuracy statistics

**Diffusion Testing**:
- Generated latent representations saved as `.npy` files
- Generation quality metrics (MSE, MAE)

***

## 6.Reference

- **Paper**: [AI-assisted spatiotemporal turbulence generation: CoNFILD](https://doi.org/10.1038/s41467-024-54712-1)
- **Authors**: Pan Du, Meet Hemant Parikh, Xiantao Fan, Xin-Yang Liu, Jian-Xun Wang
- **Original Code**: [github.com/jx-wang-s-group/CoNFILD](https://github.com/jx-wang-s-group/CoNFILD)
