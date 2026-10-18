#!/bin/bash
# Run R rounds of one arm of the self-generation loop on one GPU (generation, re-read, train/eval/amplify).
#   GPU=0 scripts/run_loop.sh ARM ROUNDS [N]
# ARM: prompt (no steering, fixed seeds) | seeds (anti-refusal, seed amplification) | full (anti-refusal, seeds and style vectors)
# N: writer prompts per behavior per round (default 100). Output: data/loop/arms/ARM/. Finished stages are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1
ARM=$1; R=$2; N=${3:-100}; G=${GPU:-0}; JE=${JUDGE_EFFORT:-low}
A=data/loop/arms/$ARM; mkdir -p $A
PY=.venv/bin/python
case $ARM in
  prompt) GEN="--no-anti-refusal"; AMP=none; K=1 ;;
  seeds)  GEN="";                  AMP=seeds; K=2 ;;
  full)   GEN="";                  AMP=full; K=3 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
for r in $(seq 0 $((R - 1))); do
  S=data/loop/seeds_r0.jsonl; [ "$AMP" != none ] && [ -f $A/seeds_r$r.jsonl ] && S=$A/seeds_r$r.jsonl
  V=""; [ "$AMP" = full ] && [ -f $A/vectors_r$r.pt ] && V="--vectors $A/vectors_r$r.pt"
  echo "$(TZ=America/New_York date +%H:%M) $ARM round $r: seeds $S ${V:-no vectors}"
  [ -f $A/r$r/candidates.jsonl ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/loop_generate.py --out $A/r$r --seeds $S $V $GEN \
      --n $N --tag "$ARM:r$r:" --seed $((10 * r + K)) --judge-effort $JE > $A/r$r.gen.log 2>&1
  [ -f $A/r$r/candidates.feats.npz ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/loop_features.py $A/r$r/candidates.jsonl > $A/r$r.feat.log 2>&1
  [ -f $A/r$r/metrics.json ] || $PY scripts/loop_round.py --arm-dir $A --round $r --amplify $AMP > $A/r$r.round.log 2>&1
  grep -E "AUROC|amplify|training data" $A/r$r.round.log || true
done
