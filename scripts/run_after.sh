#!/usr/bin/env bash
# Wait for a running overnight chain (run_refusal_overnight.sh) to finish, then run another pipeline with the
# same arguments and environment (GPU=..., PIPELINE=..., STAGES=...).
#   GPU=1 PIPELINE=steer_v1_extra.py ./scripts/run_after.sh
cd "$(dirname "$0")/.."
while pgrep -f "^bash ./scripts/run_refusal_overnight.sh" >/dev/null; do sleep 60; done
exec ./scripts/run_refusal_overnight.sh "$@"
