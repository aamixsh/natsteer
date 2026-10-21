#!/bin/bash
# Queue stand-alone round-0 runs (scripts/run_round0.sh) on free GPUs: one worker per GPU (GPUS, default "0 1") takes
# the next unclaimed, unfinished run when its GPU holds less than FREE_MB MiB (default 2000). A claim is an atomic
# mkdir ARM_DIR.claim; the run seed is the number at the end of ARM_DIR (.../s3 -> 3).
#   GEN_ARGS="--agentic-frac 0.5 --judge remote --min-realism 2" nohup scripts/queue_round0.sh SEEDS ARM_DIR... &
set -uo pipefail
cd "$(dirname "$0")/.."
SEEDS=$1; shift; JOBS="$*"
now() { TZ=America/New_York date +%H:%M; }
gpu_free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "${FREE_MB:-2000}" ]; }
left() { for e in $JOBS; do [ -f $e/r0/metrics_c3e-4.json ] || [ -d $e.claim ] || echo $e; done; }
worker() {
  local g=$1
  for d in $JOBS; do
    [ -f $d/r0/metrics_c3e-4.json ] && continue
    until gpu_free $g; do [ -n "$(left)" ] || return 0; sleep 60; done
    mkdir -p $(dirname $d) && mkdir $d.claim 2>/dev/null || continue
    echo "$(now) GPU $g: $d"
    GPU=$g scripts/run_round0.sh $d $SEEDS ${d##*s} || echo "$(now) GPU $g: $d FAILED"
  done
}
for g in ${GPUS:-0 1}; do worker $g & done
wait
echo "$(now) queue done"
