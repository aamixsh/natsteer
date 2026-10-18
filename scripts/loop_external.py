"""External-writer arm: monitors trained on misalignment-indicators transcripts (Claude Sonnet) at given budgets.

  loop_external.py --sizes 400 800 1200 1600 2000      (needs data/loop/arms/external/pool.feats.npz; see --sample)
  loop_external.py --sample 2000                       writes the nested sample pool.jsonl (re-read it with loop_features.py)

The pool is a fixed random order of data/loop/external.jsonl, balanced over behaviors; budget n uses its first n
conversations, so budgets are nested like the loop's rounds. Writes data/loop/arms/external/n<size>/metrics.json.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from natsteer.config import DATA_ROOT
from natsteer.convo import load_jsonl, save_jsonl
from natsteer.monitor import evaluate, load, train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sample", type=int, default=None)
ap.add_argument("--sizes", type=int, nargs="*", default=[])
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--eval", nargs="*", default=["own", "indicators_s", "bloom_val", "bloom_test"])
ap.add_argument("--metrics-name", default="metrics.json")
args = ap.parse_args()

A = DATA_ROOT / "loop" / "arms" / "external"
A.mkdir(parents=True, exist_ok=True)
pool_path = A / "pool.jsonl"
if args.sample:
    rows = load_jsonl(DATA_ROOT / "loop" / "external.jsonl")
    rng = random.Random(0)
    by = defaultdict(list)
    for r in rows:
        if any(t is not None for t in r["turn_labels"] or []):
            by[r["behavior"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    order = []
    while len(order) < args.sample and any(by.values()):  # round-robin over behaviors
        for b in sorted(by):
            if by[b]:
                order.append(by[b].pop())
    save_jsonl(pool_path, order[: args.sample])
    print(f"wrote {pool_path}: {len(order[: args.sample])} conversations")

if args.sizes:
    fs = load([pool_path], args.layer)
    evals = {n: load([DATA_ROOT / "loop" / "eval" / f"{n}.jsonl"], args.layer) for n in args.eval
             if (DATA_ROOT / "loop" / "eval" / f"{n}.feats.npz").exists()}
    for n in args.sizes:
        mask = (fs.w_conv < n) & ~np.isnan(fs.y)
        m = train(fs.X[mask], fs.y[mask], args.C)
        out = {"size": n, "layer": args.layer, "C": args.C, "eval": {k: evaluate(v, m) for k, v in evals.items()}}
        d = A / f"n{n}"
        d.mkdir(exist_ok=True)
        (d / args.metrics_name).write_text(json.dumps(out, indent=1))
        print(n, {k: round(v["auroc"], 3) for k, v in out["eval"].items()}, {k: round(v["tpr@1"], 3) for k, v in out["eval"].items()}, flush=True)
