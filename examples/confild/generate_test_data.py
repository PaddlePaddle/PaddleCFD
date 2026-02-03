import os

import numpy as np


# Create data directory
os.makedirs("data/Case1", exist_ok=True)

print("Generating test data for Case1 (elbow flow)...")

# Generate minimal test data
# Case1: 2D elbow flow
# Shape: [N_samples, height, width, features]
# Features: u, v, p
N_samples = 100  # Minimal samples for testing
height = 32
width = 32
features = 3  # u, v, p

# Generate synthetic flow field data
np.random.seed(42)
x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)
X, Y = np.meshgrid(x, y)

case1_data = []
for i in range(N_samples):
    # Simulate simple flow patterns
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y + i * 0.1)
    v = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y + i * 0.1)
    p = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Stack features
    flow_field = np.stack([u, v, p], axis=-1)
    case1_data.append(flow_field)

case1_data = np.array(case1_data, dtype=np.float32)
print(f"Case1 data shape: {case1_data.shape}")

# Add one dummy sample at the beginning (as load_elbow_flow skips first sample)
dummy_sample = np.zeros_like(case1_data[0:1])
case1_data = np.concatenate([dummy_sample, case1_data], axis=0)
print(f"Case1 data shape (with dummy): {case1_data.shape}")

# Save data
np.save("data/Case1/case1_data.npy", case1_data)
print("Saved: data/Case1/case1_data.npy")

# Generate coordinates
coords = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
print(f"Coordinates shape: {coords.shape}")
np.save("data/Case1/case1_coords.npy", coords)
print("Saved: data/Case1/case1_coords.npy")

print("\n" + "=" * 50)
print("Test data generation completed!")
print("=" * 50)
print(f"Total samples: {N_samples}")
print(f"Spatial resolution: {height}x{width}")
print(f"Features: {features} (u, v, p)")
print(f"Train samples: {int(N_samples * 0.7)}")
print(f"Val samples: {int(N_samples * 0.15)}")
print(f"Test samples: {N_samples - int(N_samples * 0.7) - int(N_samples * 0.15)}")
