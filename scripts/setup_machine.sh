#!/usr/bin/env bash
# One-time setup on a new machine. Idempotent; safe to rerun.
#   ./scripts/setup_machine.sh [--no-prefetch] [--skip-grader] [--skip-tests]
# 1. checks the machine (scripts/check_machine.sh), 2. installs uv if missing, 3. builds .venv from uv.lock,
# 4. downloads models, SAEs, datasets and the refusal prompt sets (scripts/prefetch.py), 5. rebuilds the
# deterministic prompt pools, 6. runs the CPU tests. Results and sampled data come from scripts/pull_data.sh.
# See docs/HANDOFF.md for the full new-machine runbook.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
PREFETCH=1; GRADER=""; TESTS=1
for a in "$@"; do
  case "$a" in
    --no-prefetch) PREFETCH=0 ;;
    --skip-grader) GRADER="--skip-grader" ;;
    --skip-tests) TESTS=0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
mkdir -p "$NATSTEER_DATA"/{runs,pools,models,checkpoints,prompts}

./scripts/check_machine.sh || { echo "fix the failures above, then rerun"; exit 1; }

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$NATSTEER_BIN" UV_NO_MODIFY_PATH=1 sh
  export PATH="$NATSTEER_BIN:$PATH"
fi
uv sync --locked

if [ "$PREFETCH" = 1 ]; then
  .venv/bin/python scripts/prefetch.py $GRADER
  # Prompt pools are deterministic and take seconds; sampled data (syco_v0, refusal_v0) comes from pull_data.sh.
  [ -f "$NATSTEER_DATA/pools/intent_v0/human_eval.jsonl" ] || .venv/bin/python scripts/build_intent_pool.py
  [ -f "$NATSTEER_DATA/pools/aegis_v0/train.jsonl" ] || .venv/bin/python scripts/build_aegis_pool.py
fi

[ "$TESTS" = 1 ] && .venv/bin/python -m pytest -q tests
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
echo "setup done. Next: ./scripts/pull_data.sh <source>  (optional), then ./scripts/verify_machine.sh"
