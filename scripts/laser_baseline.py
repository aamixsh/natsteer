"""LASER vs random vs uncertainty-only on natural data, plus all-label probe sweeps.

Labels come from the rows' human_label, or from grader files (--pool-grades/--eval-grades).
--pos-rate subsamples pool positives to a target base rate (fixed by --pool-seed), to put
the pool in LASER's rare-positive regime.

Usage:
  .venv/bin/python scripts/laser_baseline.py --name aegis_human \
      --pool data/pools/aegis_v0/train.jsonl --eval data/pools/intent_v0/human_eval.jsonl \
      --eval-source aegis --pos-rate 0.02
"""

import argparse
import json
from pathlib import Path

import numpy as np

from natsteer import laser
from natsteer.config import run_dir
from natsteer.probes import auroc, bootstrap_ci, make_probe, tpr_at_fpr


def load(rows_path, grades_path=None, source=None):
    rows = [json.loads(l) for l in open(rows_path)]
    feats = np.load(Path(rows_path).with_suffix(".feats.npz"))
    assert list(feats["ids"]) == [r["id"] for r in rows]
    if grades_path:
        g = {x["id"]: x["label"] for x in map(json.loads, open(grades_path))}
        y = np.array([g.get(r["id"]) if g.get(r["id"]) is not None else -1 for r in rows])
    else:
        y = np.array([r["human_label"] if r.get("human_label") is not None else -1 for r in rows])
    keep = y >= 0
    for k in feats.files:  # a few over-long prompts were truncated and have no prompt_last
        if k != "ids":
            keep &= np.isfinite(feats[k]).all(1)
    if source:
        keep &= np.array([r["source"] == source for r in rows])
    return feats, y, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--pool-grades")
    ap.add_argument("--eval-grades")
    ap.add_argument("--pool-source")
    ap.add_argument("--eval-source")
    ap.add_argument("--pos-rate", type=float)
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--feature", default="L15_user_mean")
    ap.add_argument("--arms", nargs="+", default=["laser", "uncertainty", "random"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--init", type=int, default=100)
    args = ap.parse_args()

    pf, py, pkeep = load(args.pool, args.pool_grades, args.pool_source)
    ef, ey, ekeep = load(args.eval, args.eval_grades, args.eval_source)
    idx = np.flatnonzero(pkeep)
    if args.pos_rate:
        rng = np.random.default_rng(args.pool_seed)
        pos, neg = idx[py[idx] == 1], idx[py[idx] == 0]
        n_pos = int(round(args.pos_rate * len(neg) / (1 - args.pos_rate)))
        idx = np.sort(np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)]))
    y_pool, y_eval = py[idx], ey[ekeep]
    out = run_dir(f"laser_baseline/{args.name}")
    report = {"args": vars(args), "pool_n": int(len(idx)), "pool_pos": int(y_pool.sum()), "eval_n": int(len(y_eval)), "eval_pos": int(y_eval.sum())}
    print(report, flush=True)

    # All-label probes per layer and pooling: the ceiling for any label-efficient method.
    sweep = {}
    for key in sorted(k for k in pf.files if k != "ids"):
        Xp, Xe = pf[key][idx].astype(np.float32), ef[key][ekeep].astype(np.float32)
        s = make_probe().fit(Xp, y_pool).decision_function(Xe)
        sweep[key] = {"auroc": auroc(y_eval, s), "tpr@0.01": tpr_at_fpr(y_eval, s, 0.01), "tpr@0.05": tpr_at_fpr(y_eval, s, 0.05), "tpr@0.01_ci": bootstrap_ci(lambda a, b: tpr_at_fpr(a, b, 0.01), y_eval, s, n=300)}
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sweep[key].items()}, flush=True)
    report["all_label_sweep"] = sweep

    Xp, Xe = pf[args.feature][idx].astype(np.float32), ef[args.feature][ekeep].astype(np.float32)
    curves = {}
    for arm in args.arms:
        curves[arm] = [laser.run(Xp, lambda i: y_pool[i], Xe, y_eval, budget=args.budget, batch=args.batch, init=args.init, strategy=arm, seed=s) for s in range(args.seeds)]
        last = [c[-1] for c in curves[arm]]
        print(arm, "final:", {k: round(float(np.mean([p.get(k, np.nan) for p in last])), 3) for k in ("n_pos", "auroc", "tpr@0.01", "tpr@0.05")}, flush=True)
    report["curves"] = curves
    json.dump(report, open(out / "report.json", "w"), indent=1)
    plot(curves, out / "curves.png", f"{args.name}: {args.feature}, pool {len(idx)} ({int(y_pool.sum())} pos)")
    print("saved", out, flush=True)


def plot(curves, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("tpr@0.01", "auroc", "n_pos")):
        for arm, runs in curves.items():
            x = [p["n_labels"] for p in runs[0]]
            v = np.array([[p.get(metric, np.nan) for p in run] for run in runs], dtype=float)
            m, sd = np.nanmean(v, 0), np.nanstd(v, 0)
            ax.plot(x, m, label=arm)
            ax.fill_between(x, m - sd, m + sd, alpha=0.2)
        ax.set_xlabel("labels")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
