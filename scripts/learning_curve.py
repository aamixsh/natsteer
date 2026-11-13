"""(CPU; run from the repo root.) Learning curve: does more writer data, or more varied writer data, help the monitor?

Training pools (accepted writer conversations with layer-15 features; a positive and its aligned twin share a group
and are drawn together):
  scale        arms/scale_v1 rounds only (one fixed pipeline: v3 seeds, branch negatives, gpt-6.1-sol, checklist)
  sol_pooled   branch_sol + agentic_sol + scale_v1 (gpt-6.1-sol labels, several pipelines)
  all_pooled   sol_pooled + the local-judge rounds of the prompt, seeds and full arms (rounds 1-4 for seeds and
               full: their round 0 writes are the branch_sol writes)
For each pool and size N (conversations) it trains --draws monitors (StandardScaler + logistic regression, C 3e-4,
windows labeled by their turn) on random groups up to N, and scores Bloom test, own rollouts without the seed
scenarios and, if its features exist, the checkable sandbagging test: AUROC, detection at 1% and per-behavior
detection. Same size, different pools = the variety effect; same pool, growing size = the size effect.
Output: docs/results/learning_curve_v0/report.json and a printed table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from natsteer.convo import load_jsonl
from natsteer.monitor import evaluate_scores, feats_path, load, train, window_scores

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000, 8000, 16000])
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--C", type=float, default=3e-4)
ap.add_argument("--jobs", type=int, default=40)
ap.add_argument("--out", default="docs/results/learning_curve_v0/report.json")
args = ap.parse_args()

A, L = Path("data/loop/arms"), Path("data/loop")


def runs(pattern):
    return sorted(p for p in A.glob(pattern) if feats_path(p).exists())


SRC = {"branch_sol": runs("branch_sol/s*/r0/candidates.jsonl"), "agentic_sol": runs("agentic_sol/s*/r0/candidates.jsonl"),
       "scale_v1": runs("scale_v1/s*/r0/candidates.jsonl"),
       "local": runs("prompt/r[0-4]/candidates.jsonl") + runs("seeds/r[1-4]/candidates.jsonl") + runs("full/r[1-4]/candidates.jsonl")}
POOLS = {"scale": ["scale_v1"], "sol_pooled": ["branch_sol", "agentic_sol", "scale_v1"], "all_pooled": ["branch_sol", "agentic_sol", "scale_v1", "local"]}
print({k: len(v) for k, v in SRC.items()}, flush=True)

paths = [p for k in SRC for p in SRC[k]]
src_of_path = {p: k for k in SRC for p in SRC[k]}
fs = load(paths, 15)
conv_src, conv_run = [], []
for p in paths:
    n = len(load_jsonl(p))
    conv_src += [src_of_path[p]] * n
    conv_run += [str(p.parent)] * n
conv_src, conv_run = np.array(conv_src), np.array(conv_run)
# groups are unique per run (job ids repeat across runs)
gkey = np.array([f"{r}|{c['group']}" for r, c in zip(conv_run, fs.convs)])
lab = ~np.isnan(fs.y)
print(f"training conversations {len(fs.convs)}, labeled windows {int(lab.sum())}", flush=True)


def held_set():
    h = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], 15)
    seeds = {s["group"] for s in load_jsonl(L / "seeds_onpolicy.jsonl") if s["source"] == "onpolicy"}
    for c in h.convs:
        if c["group"] in seeds:
            c["turn_labels"] = None
    return h


tests = {"bloom_test": load([L / "eval" / "bloom_test.jsonl"], 15), "own": held_set()}
if feats_path(Path("data/sandbag_math/test.jsonl")).exists():
    tests["sandbag_math"] = load([Path("data/sandbag_math/test.jsonl")], 15)


def fit_eval(pool, size, draw):
    rng = np.random.default_rng(1000 * size + draw)
    in_pool = np.isin(conv_src, POOLS[pool])
    groups = np.unique(gkey[in_pool])
    rng.shuffle(groups)
    sizes = {g: 0 for g in groups}
    for g in gkey[in_pool]:
        sizes[g] += 1
    take, n = [], 0
    for g in groups:
        if n >= size:
            break
        take.append(g)
        n += sizes[g]
    convs = np.flatnonzero(np.isin(gkey, take))
    w = fs.subset(convs) & lab
    m = train(fs.X[w], fs.y[w], args.C)
    res = {"pool": pool, "size": size, "draw": draw, "n_conv": int(len(convs)), "n_pos": int(sum(fs.convs[c]["label"] == 1 for c in convs)),
           "n_runs": int(len(np.unique(conv_run[convs]))), "n_windows": int(w.sum())}
    for name, t in tests.items():
        e = evaluate_scores(t, window_scores(m, t.X), n_boot=0)
        res[name] = {"auroc": e["auroc"], "tpr@1": e["tpr@1"], "per_behavior_tpr@1": {b: v["tpr"] for b, v in e["per_behavior_tpr@1"].items()}}
    return res


jobs = []
for pool in POOLS:
    avail = int(np.isin(conv_src, POOLS[pool]).sum())
    if avail == 0:
        continue
    for size in [s for s in args.sizes if s < avail] + [avail]:
        for draw in range(1 if size == avail else args.draws):
            jobs.append((pool, size, draw))
print(f"{len(jobs)} fits", flush=True)
results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(fit_eval)(*j) for j in jobs)

summary = {}
for r in results:
    s = summary.setdefault(r["pool"], {}).setdefault(str(r["size"]), {"n_conv": [], **{t: {"auroc": [], "tpr@1": []} for t in tests}})
    s["n_conv"].append(r["n_conv"])
    for t in tests:
        for k in ("auroc", "tpr@1"):
            s[t][k].append(r[t][k])
for pool, by in summary.items():
    for size, s in by.items():
        s["n_conv"] = float(np.mean(s["n_conv"]))
        for t in tests:
            for k in ("auroc", "tpr@1"):
                v = s[t][k]
                s[t][k] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"sources": {k: [str(p) for p in v] for k, v in SRC.items()}, "pools": POOLS, "fits": results, "summary": summary}, indent=1))
print("\nmean (SD) over draws: AUROC / detection at 1%")
for pool, by in summary.items():
    for size, s in sorted(by.items(), key=lambda x: int(x[0])):
        print(f"{pool:11s} N={s['n_conv']:7.0f}  " + "  ".join(f"{t} {s[t]['auroc'][0]:.3f} ({s[t]['auroc'][1]:.3f}) / {s[t]['tpr@1'][0]:.2f}" for t in tests))
