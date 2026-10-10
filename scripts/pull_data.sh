#!/usr/bin/env bash
# Copy results and sampled data from another machine. Everything rebuildable is skipped.
#   ./scripts/pull_data.sh user@host:/path/to/natsteer [--big] [--dry-run]
# or from a local path (a mounted disk, an unpacked archive): ./scripts/pull_data.sh /mnt/old/natsteer
# Copies data/{runs,pools,handoff,prompts}. Without --big the pooled features (*.feats.npz, about 8 GB)
# are skipped; they can be rebuilt (scripts/extract_features.py, syco_pipeline.py --stage features). Pools'
# jsonl files built by build_*_pool.py are small and copied too. What cannot be rebuilt bit for bit:
# data/runs (reports), data/pools/syco_v0 and data/pools/refusal_v0 (sampled generations).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
SRC="${1:?usage: pull_data.sh user@host:/path/to/natsteer [--big] [--dry-run]}"; shift || true
EXCL=(--exclude='*.feats.npz'); DRY=()
for a in "$@"; do
  case "$a" in
    --big) EXCL=() ;;
    --dry-run) DRY=(--dry-run) ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
SRC="${SRC%/}"
for d in runs pools handoff prompts; do
  rsync -avP "${DRY[@]}" "${EXCL[@]}" "$SRC/data/$d/" "$NATSTEER_DATA/$d/" || echo "note: $SRC/data/$d not copied"
done
echo "pull done into $NATSTEER_DATA"
