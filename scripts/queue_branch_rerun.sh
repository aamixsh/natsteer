#!/bin/bash
# Branch rerun with gpt-6.1-sol as judge (PLAN next todo 1). Writes are copied from the three Claude-seed round-0
# runs of monitor_v0 (seed 2: arms/seeds/r0, 3: arms/full/r0, 4: arms/seedsrc/claude_s4/r0), so the positives'
# text is fixed and only the judge and the negatives change.
#   branch_sol/s{2,3,4}:  gpt-6.1-sol judges the writes; branch from its first misaligned turn; judge the branches
#   rewrite_sol/s{2,3,4}: the old rewrites judged by gpt-6.1-sol; the positives' judgments shared with branch_sol
# Positives need realism >= 2 under gpt-6.1-sol (its realism scale is harsher: at >= 3 it keeps 138 of 449 writes and
# almost no sabotage or sandbagging; at >= 2, 295, close to the local judge's 267 at >= 3).
# Step 1 (no GPU, API only) judges the writes and the rewrites now. Step 2 runs the GPU stages: one worker per GPU
# takes the next unclaimed job when its GPU holds less than FREE_MB MiB (default 2000). Step 3 writes the report (CPU).
#   scripts/queue_branch_rerun.sh               # all steps; run it in the background (nohup ... &)
#   PREP_ONLY=1 scripts/queue_branch_rerun.sh   # step 1 only
# Spend: data/loop/arms/sol_spend.jsonl, hard cap CAP (default $30) on the ledger total.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1
PY=.venv/bin/python; A=data/loop/arms; SEEDS=data/loop/seeds_r0.jsonl
J="--judge remote --judge-cap ${CAP:-30} --min-realism 2"
declare -A SRC=([2]=$A/seeds/r0 [3]=$A/full/r0 [4]=$A/seedsrc/claude_s4/r0)
now() { TZ=America/New_York date +%H:%M; }

for s in 2 3 4; do
  B=$A/branch_sol/s$s; R=$A/rewrite_sol/s$s
  mkdir -p $B/r0 $R/r0
  [ -f $B/r0/writes.jsonl ] || cp ${SRC[$s]}/writes.jsonl $B/r0/
  [ -f $R/r0/writes.jsonl ] || cp ${SRC[$s]}/writes.jsonl $R/r0/
  [ -f $R/r0/rewrites.jsonl ] || cp ${SRC[$s]}/rewrites.jsonl $R/r0/
  echo "$(now) prep s$s"
  [ -f $B/r0/judged_pos.jsonl ] || CUDA_VISIBLE_DEVICES="" $PY scripts/loop_generate.py --out $B/r0 --seeds $SEEDS --seed $s $J --judge-only > $B/r0.prep.log 2>&1
  [ -f $R/r0/candidates.jsonl ] || CUDA_VISIBLE_DEVICES="" $PY scripts/loop_generate.py --out $R/r0 --seeds $SEEDS --seed $s --negatives rewrite $J \
      --pos-judgments $B/r0/judged_pos.jsonl > $R/r0.prep.log 2>&1
done
echo "$(now) prep done; spend so far \$$($PY -c "from natsteer.remote import ledger_total; print(round(ledger_total('$A/sol_spend.jsonl'), 2))")"
[ -n "${PREP_ONLY:-}" ] && exit 0

JOBS="$A/branch_sol/s2 $A/branch_sol/s3 $A/branch_sol/s4 $A/rewrite_sol/s2 $A/rewrite_sol/s3 $A/rewrite_sol/s4"
gpu_free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "${FREE_MB:-2000}" ]; }
worker() {  # takes unclaimed, unfinished jobs while any remain; a claim is an atomic mkdir
  local g=$1
  for d in $JOBS; do
    [ -f $d/r0/metrics_c3e-4.json ] && continue
    until gpu_free $g; do
      [ -n "$(for e in $JOBS; do [ -f $e/r0/metrics_c3e-4.json ] || [ -d $e.claim ] || echo $e; done)" ] || return 0
      sleep 60
    done
    mkdir $d.claim 2>/dev/null || continue
    echo "$(now) GPU $g: $d"
    GPU=$g GEN_ARGS="$J" scripts/run_round0.sh $d $SEEDS ${d##*/s} || echo "$(now) GPU $g: $d FAILED"
  done
}
worker 0 & worker 1 &
wait
echo "$(now) GPU stages done"
$PY scripts/branch_rerun_report.py
echo "$(now) all done"
