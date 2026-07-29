#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 单一开关：train.py 据此在 import paddle 之前设 CINN FLAGS 并对 model 做 to_static。
export POSEIDON_USE_CINN=1

export WANDB_MODE=disabled
# No PYTHONPATH hack needed — examples/poseidon uses relative / installed-package imports.

cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/train.py" \
  --config "${SCRIPT_DIR}/configs/run_small.yaml" \
  --wandb_run_name "se-af-cinn" \
  --wandb_project_name "PaddleCFD-Poseidon" \
  --checkpoint_path "${REPO_ROOT}/tmp_checkpoints" \
  --data_path "${SCRIPT_DIR}/../dataset"
