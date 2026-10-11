#!/usr/bin/env bash
# Overnight steering pipelines on one GPU. Every stage skips if its output exists, so rerunning resumes.
#   GPU=1 ./scripts/run_refusal_overnight.sh                  first refusal pass (refusal_pipeline.py), outputs in data/pools/refusal_v0
#   GPU=1 PIPELINE=steer_v1.py ./scripts/run_refusal_overnight.sh   second pass: refusal v1 and sycophancy, data/pools/steer_v1
#   GPU=1 PIPELINE=steer_v1_extra.py ./scripts/run_after.sh   extras (reachability, timing, LASER warm start) once the running chain ends
#   add --smoke for a tiny run (outputs in data/pools/<name>_smoke)
#   GPU=1 STAGES="latents generate" ./scripts/run_refusal_overnight.sh --smoke   only these stages
# Watchdog: a stage with no log output for IDLE seconds (default 1200) is killed with its whole
# process group (vLLM engine included) and retried, up to 3 attempts. A vLLM engine once hung at
# start-up here (2026-10-10, waiting on a lock before loading weights), so this is not hypothetical.
# A failed stage is logged and the chain continues; later stages that need its output fail fast.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU, e.g. GPU=1}"
EXTRA="${1:-}"
PIPELINE="${PIPELINE:-refusal_pipeline.py}"
if [ "$PIPELINE" = "steer_v1.py" ]; then
  OUTDIR=data/pools/steer_v1; SMOKE_DIR=data/pools/steer_v1_smoke
  DEFAULT_STAGES="init natural_more attribution_wide latents_screen latents_validate generate features probes syco_directions syco_attr_wide syco_sweep syco_latents syco_generate syco_features syco_probes report"
elif [ "$PIPELINE" = "steer_v1_extra.py" ]; then
  OUTDIR=data/pools/steer_v1; SMOKE_DIR=data/pools/steer_v1_smoke
  DEFAULT_STAGES="laser_warmstart probes_wildchat reach timing_hf summary report_extra"
else
  OUTDIR=data/pools/refusal_v0; SMOKE_DIR=data/pools/refusal_smoke
  DEFAULT_STAGES="natural directions sweep attribution latents generate features probes sweep_hf syco_natural syco_attr report"
fi
[ "$EXTRA" = "--smoke" ] && OUTDIR=$SMOKE_DIR
IDLE="${IDLE:-900}"
STAGES="${STAGES:-$DEFAULT_STAGES}"
mkdir -p "$OUTDIR/logs"
TIMINGS="$OUTDIR/timings.log"

run_stage() {
  local stage=$1 attempt rc=1
  for attempt in 1 2 3; do
    local slog="$OUTDIR/logs/$stage.$attempt.log"
    setsid .venv/bin/python scripts/"$PIPELINE" --stage "$stage" $EXTRA > "$slog" 2>&1 &
    local pid=$! last=-1 idle=0 size
    tail -n +1 -f --pid=$pid "$slog" &
    local tailpid=$!
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      size=$(stat -c %s "$slog")
      if [ "$size" = "$last" ]; then idle=$((idle + 30)); else idle=0; last=$size; fi
      if [ "$idle" -ge "$IDLE" ]; then
        echo "WATCHDOG $stage attempt $attempt: no output for ${IDLE}s, killing process group $pid" | tee -a "$TIMINGS"
        kill -TERM -- "-$pid" 2>/dev/null; sleep 30; kill -KILL -- "-$pid" 2>/dev/null
        break
      fi
    done
    wait "$pid"; rc=$?
    kill "$tailpid" 2>/dev/null
    [ "$rc" -eq 0 ] && return 0
    echo "$stage attempt $attempt failed rc=$rc" | tee -a "$TIMINGS"
  done
  return "$rc"
}

for stage in $STAGES; do
  start=$(date +%s)
  echo "=== $(date '+%F %T') $stage (GPU $CUDA_VISIBLE_DEVICES)"
  run_stage "$stage"
  rc=$?
  echo "$stage rc=$rc $(( $(date +%s) - start ))s" | tee -a "$TIMINGS"
done
echo "=== $(date '+%F %T') chain finished"
