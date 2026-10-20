#!/bin/bash
# Style-vector strength sweep: one vector per behavior (largest cluster of a vectors file), every prompt styled,
# at scales 0 (no vector) and x1, x2, x4, x8 of the stored strength (alpha 0.1 x median residual norm).
#   GPU=1 scripts/style_sweep.sh [VECTORS] [N] [SCALES...]
# Output: data/loop/style_sweep/s<scale>/ (generation + re-read); then scripts/style_sweep_report.py.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1
V=${1:-data/loop/arms/full/vectors_r5.pt}; N=${2:-30}; shift 2 || true
SCALES=${@:-"0 1 2 4 8"}
G=${GPU:-0}; PY=.venv/bin/python; O=data/loop/style_sweep
mkdir -p $O
$PY - "$V" "$O/vectors.pt" <<'EOF'
import sys, torch
cards = torch.load(sys.argv[1])
best = {}
for c in cards:
    if c["behavior"] not in best or c["n"] > best[c["behavior"]]["n"]:
        best[c["behavior"]] = c
torch.save(list(best.values()), sys.argv[2])
print("vectors:", [c["id"] for c in best.values()])
EOF
for s in $SCALES; do
  D=$O/s$s
  if [ "$s" = "0" ]; then VEC=""; else VEC="--vectors $O/vectors.pt --style-frac 1.0 --style-scale $s"; fi
  echo "$(TZ=America/New_York date +%H:%M) scale $s"
  [ -f $D/candidates.jsonl ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/loop_generate.py --out $D --seeds data/loop/seeds_r0.jsonl --n $N \
      --tag "sweep_s$s:" --seed 11 $VEC > $D.gen.log 2>&1
  [ -s $D/candidates.jsonl ] || { echo "  scale $s: no accepted conversations (writer broke down)"; continue; }
  [ -f $D/candidates.feats.npz ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/loop_features.py $D/candidates.jsonl > $D.feat.log 2>&1
done
$PY scripts/style_sweep_report.py
