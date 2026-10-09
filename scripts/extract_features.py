"""Pooled clean activations for every row of a jsonl file of user prompts.

Renders each prompt alone (harmony, fixed date, effort medium), runs gpt-oss-20b in
bf16, and stores user_mean and prompt_last at each SAE layer as float16.
Output: <out>.npz with feats[layer][pooling] -> [n, d_model] and ids.
Usage: .venv/bin/python scripts/extract_features.py data/pools/intent_v0/pool.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from natsteer.config import SAE_LAYERS
from natsteer.features import extract, save
from natsteer.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--layers", type=int, nargs="+", default=list(SAE_LAYERS))
    args = ap.parse_args()
    model, tok = load_model("bf16")
    for path in map(Path, args.inputs):
        rows = [json.loads(l) for l in open(path)]
        t0 = time.time()
        out = path.with_suffix(".feats.npz")
        save(out, rows, extract(model, tok, rows, args.layers))
        print(f"saved {out} ({len(rows)} rows, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
