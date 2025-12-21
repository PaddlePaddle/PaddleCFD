# Graph Networks for Physics Discovery

## 1.Background

Graph Neural Networks (GNNs) can learn and discover underlying physical laws from particle interaction data. This framework combines the powerful fitting capabilities of deep learning with the interpretability of symbolic regression, enabling automatic discovery of physical equations such as Newtonian mechanics laws and Hamiltonian systems.

### Key Features

- **Three Model Architectures**: OGN (Newtonian mechanics), HGN (Hamiltonian mechanics), VarOGN (with uncertainty quantification)
- **Multiple Physical Systems**: Supports 9 physical systems including springs, gravity, charges, and damping
- **Symbolic Regression Ready**: L1 regularization encourages sparse representations for subsequent symbolic regression
- **Modular Design**: Model code in `ppcfd/models/symbolic_gn`, training/inference code in `examples/symbolic_gn`

***

## 2.Model Description

### 2.1 OGN (Object-based Graph Network)

**Physical Foundation**: Newtonian mechanics (F = ma)

**Core Concept**:
- Models particle systems as graphs: nodes = particles, edges = interactions
- Message passing mechanism: Computes force messages between particles
- Node update: Predicts acceleration based on aggregated force messages

**Network Structure**:
```
Particle features [x, y, vx, vy, charge, mass]
    ↓
Message function φₑ(xᵢ, xⱼ) → Message vector (100-dim)
    ↓
Message aggregation Σ mᵢⱼ
    ↓
Node update φᵥ(xᵢ, Σmᵢⱼ) → Acceleration [ax, ay]
```

**Application Scenarios**: General dynamical systems (springs, gravity, charges, etc.)

### 2.2 HGN (Hamiltonian Graph Network)

**Physical Foundation**: Hamiltonian mechanics (energy conservation)

**Core Concept**:
- Learns system Hamiltonian H = Σ Eᵢ + Σ Eᵢⱼ
- Implements Hamilton's equations via automatic differentiation:
  - dq/dt = ∂H/∂p (generalized velocity)
  - dp/dt = -∂H/∂q (generalized force)

**Network Structure**:
```
Particle state [q, v, charge, mass] → [q, p=mv, charge, mass]
    ↓
Pairwise energy Eᵢⱼ = φₑ(xᵢ, xⱼ)
Self energy Eᵢ = φᵥ(xᵢ)
    ↓
Hamiltonian H = Σ Eᵢⱼ + Σ Eᵢ
    ↓
Auto-differentiation ∇H → [dq/dt, dp/dt] → [v, a]
```

**Application Scenarios**: Conservative systems (no dissipation), long-term prediction

### 2.3 VarOGN (Variational OGN)

**Physical Foundation**: Variational inference + Newtonian mechanics

**Core Concept**:
- Message function outputs mean μ and log-variance logσ²
- Training: Sample m = μ + ε·exp(logσ²/2), ε ~ N(0,1)
- Inference: Use mean m = μ

**Application Scenarios**: Noisy data, uncertainty quantification, robustness evaluation

***

## 3.Supported Physical Systems

| System Type | DATA.type | Potential Function | Physical Meaning |
|-------------|-----------|-------------------|------------------|
| Gravity/Coulomb | `r2` | -m₁m₂/r | Universal gravitation/Coulomb attraction |
| 2D Gravity | `r1` | m₁m₂·log(r) | 2D vortex system |
| Spring | `spring` | (r-1)² | Hooke's law |
| Damped Spring | `damped` | (r-1)² + damping | Damped oscillation |
| String | `string` | (r-1)² + y·q | String in gravitational field |
| Charge | `charge` | q₁q₂/r | Coulomb force (repulsive/attractive) |
| Superposition | `superposition` | q₁q₂/r - m₁m₂/r | Electromagnetic + gravity |
| Discontinuous | `discontinuous` | Piecewise function | Collisions and non-smooth phenomena |
| String-Ball | `string_ball` | Spring + sphere repulsion | Complex constraints |

***

## 4.Model Training & Testing

### 4.1 Environment Setup

```bash
pip install scipy scikit-learn celluloid matplotlib
```

### 4.2 Configuration Parameters

#### Model Configuration

```yaml
MODEL:
  arch: "OGN"           # Model type: "OGN", "HGN", "VarOGN"
  msg_dim: 100          # Message dimension (OGN/VarOGN)
  hidden: 300           # Hidden layer size
  regularization_type: "l1"  # Regularization type
  l1_strength: 1e-2     # L1 regularization strength (encourages sparsity)
```

**Regularization Notes**:
- L1 regularization forces message vectors to be sparse, keeping only a few active dimensions
- Sparse representations facilitate symbolic regression for extracting physical laws
- HGN does not require L1 regularization

#### Data Configuration

```yaml
DATA:
  type: "spring"        # Physical system type
  num_samples: 1000     # Number of simulation trajectories
  num_nodes: 6          # Number of particles
  dimension: 2          # Spatial dimension (2 or 3)
  time_steps: 100       # Time steps per trajectory
  time_step_size: 0.01  # Time step size dt
  sample_interval: 5    # Training data downsampling factor
```

**Data Volume Calculation**:
- Total samples = num_samples × time_steps / sample_interval
- Example: 1000 × 100 / 5 = 20,000 training samples

#### Training Configuration

```yaml
TRAIN:
  epochs: 1000              # Training epochs
  batch_size: 32            # Batch size
  save_freq: 100            # Save frequency
  optimizer:
    learning_rate: 1e-3     # Initial learning rate
    weight_decay: 1e-8      # Weight decay
  lr_scheduler:
    name: "CosineAnnealingDecay"  # Learning rate scheduler
    gamma: 0.95             # Decay coefficient (for ExponentialDecay)
  loss:
    type: "MAE"             # Loss function: "MAE" or "MSE"
```

### 4.3 Train Models

```bash
# Train OGN model on spring system
python main.py MODEL.arch=OGN DATA.type=spring

# Train HGN model on gravity system
python main.py --config-name=config_hgn DATA.type=r2

# Train VarOGN model (with uncertainty quantification)
python main.py --config-name=config_varogn DATA.type=charge
```

### 4.4 Test Models

```bash
# Test with best model
python main.py mode=test

# Test with specific checkpoint
python main.py mode=test checkpoint="./outputs/.../OGN_best.pdparams"
```

### 4.5 Parameter Sweeps

```bash
# Test different physical systems
python main.py DATA.type=spring,r2,charge

# Test different numbers of nodes
python main.py DATA.num_nodes=4,6,8
```

***

## 5.Results

After training, the `outputs/` directory will contain:

- `{MODEL}_best.pdparams`: Best model weights
- `{MODEL}_final.pdparams`: Final model weights
- `loss_curve.png`: Training and validation loss curves
- `train.log`: Training log file

**Testing Outputs**:
- Terminal output shows MAE, MSE, RMSE metrics
- Evaluation results logged to file

**Expected Performance**:
- Spring system: Successfully learns F ∝ (r-1) (Hooke's law)
- Gravity system: Successfully learns F ∝ -m₁m₂/r² (Universal gravitation)
- Charge system: Successfully learns F ∝ q₁q₂/r² (Coulomb's law)

***

## 6.Physics Discovery Workflow

While this example focuses on graph neural network training, symbolic regression can be performed after training:

### 6.1 Train Sparse GNN

```bash
# Train OGN with L1 regularization
python main.py MODEL.l1_strength=1e-2
```

### 6.2 Extract Message Vectors (Future Work)

```python
# Save message vectors during training loop
messages = model.msg_fnc(edge_features)  # [num_edges, msg_dim]
# Save messages with corresponding physical features (dx, dy, r, m1, m2, etc.)
```

### 6.3 Symbolic Regression (Requires PySR)

```python
from pysr import PySRRegressor

# Select most active message channels
active_channels = np.argsort(np.std(messages, axis=0))[-5:]

# Perform symbolic regression on each channel
for ch in active_channels:
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "/", "-"],
        unary_operators=["square", "sqrt", "neg"],
    )
    model.fit(physical_features, messages[:, ch])
    print(f"Channel {ch}: {model.sympy()}")
```

### 6.4 Expected Discoveries

- Spring system: `F ∝ (r-1)` (Hooke's law)
- Gravity system: `F ∝ -m₁m₂/r²` (Universal gravitation)
- Charge system: `F ∝ q₁q₂/r²` (Coulomb's law)

***

## 7.Reference

- **Paper**: [Discovering Symbolic Models from Deep Learning with Inductive Biases (NeurIPS 2020)](https://arxiv.org/abs/2006.11287)
- **Authors**: Miles Cranmer et al.
- **Original Code**: [github.com/MilesCranmer/symbolic_deep_learning](https://github.com/MilesCranmer/symbolic_deep_learning)
- **Symbolic Regression Tool**: [PySR](https://github.com/MilesCranmer/PySR)
