#!/bin/bash
# One stand-alone round 0 with anti-refusal from a given seed file: write, re-read, train and evaluate (C = 3e-4).
#   GPU=0 scripts/run_round0.sh ARM_DIR SEEDS SEED [N]
# Used to compare seed sources (Claude-written vs on-policy) over replicates. Output: ARM_DIR/r0/, metrics in
# ARM_DIR/r0/metrics_c3e-4.json. Finished stages are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1
A=$1; S=$2; SEED=$3; N=${4:-100}; G=${GPU:-0}
PY=.venv/bin/python
mkdir -p $A
echo "$(TZ=America/New_York date +%H:%M) $A: seeds $S, seed $SEED"
[ -f $A/r0/candidates.jsonl ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/loop_generate.py --out $A/r0 --seeds $S --n $N \
    --tag "$(basename $A):r0:" --seed $SEED > $A/r0.gen.log 2>&1
[ -f $A/r0/candidates.feats.npz ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/loop_features.py $A/r0/candidates.jsonl > $A/r0.feat.log 2>&1
[ -f $A/r0/metrics_c3e-4.json ] || $PY scripts/loop_round.py --arm-dir $A --round 0 --C 3e-4 --metrics-name metrics_c3e-4.json > $A/r0.round.log 2>&1
grep -E "AUROC|training data" $A/r0.round.log || true
echo "$(TZ=America/New_York date +%H:%M) $A done"
