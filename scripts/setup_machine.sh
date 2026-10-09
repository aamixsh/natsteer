#!/usr/bin/env bash
# One-time setup on a new machine. Idempotent; safe to rerun.
#   1. creates data/ and scratch/ (or uses NATSTEER_DATA / NATSTEER_SCRATCH),
#   2. builds .venv from uv.lock,
#   3. downloads models, SAEs and datasets (unless --no-prefetch),
#   4. runs the CPU tests and reports the GPUs.
# Requirements: curl (uv is installed into .cache/bin if missing), an NVIDIA driver that supports CUDA 13 (driver >= 580), a Hugging
# Face login (`uv run hf auth login`) for gated content, about 60 GB of disk.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh  # project-local data, temp, caches and Python (keeps existing settings)
mkdir -p "$NATSTEER_DATA"/{runs,pools,models,checkpoints}

if ! command -v uv >/dev/null; then
  # Install uv into the project so it survives restarts of machines where only the
  # project volume persists.
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$NATSTEER_ROOT/.cache/bin" UV_NO_MODIFY_PATH=1 sh
  export PATH="$NATSTEER_ROOT/.cache/bin:$PATH"
fi
uv sync --locked

if [ "${1:-}" != "--no-prefetch" ]; then
  .venv/bin/python scripts/prefetch.py
fi

.venv/bin/python -m pytest -q tests
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
echo "setup done. Next: .venv/bin/python scripts/e0_smoke.py --keep-model"
