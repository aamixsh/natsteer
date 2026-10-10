#!/usr/bin/env bash
# Verify a set-up machine end to end on one GPU (about 10 minutes): CPU tests, the E0 smoke test compared
# with the first machine's numbers, and the vLLM steering parity check.
#   GPU=0 ./scripts/verify_machine.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU to a free GPU index, e.g. GPU=0}"
rc=0
.venv/bin/python -m pytest -q tests || rc=1
.venv/bin/python scripts/e0_smoke.py --keep-model || rc=1
.venv/bin/python scripts/compare_e0.py || rc=1
.venv/bin/python scripts/check_vllm_steering.py || rc=1
[ $rc -eq 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED (see above)"
exit $rc
