"""Report for the loop arms: detection per round, coverage of Bloom positives, diversity and style steerability.

  loop_report.py [--out docs/results/monitor_v0]

Detection: metrics.json of each arm and round (and the external arm per budget).
Coverage: for each training set (arm, rounds <= r), the median distance from every Bloom-test misaligned turn to
its nearest training misaligned turn, in a standardized PCA space of turn features fitted on all data (lower =
better covered); also the share of Bloom-test misaligned turns whose nearest training neighbour is closer than the
median within-Bloom nearest-neighbour distance.
Diversity: mean pairwise distance between training misaligned turns in the same space, and the number of
distinct (deployment, domain) settings.
Steerability (full arm): for conversations written with style vector v, the projection of (misaligned-turn
feature - mean feature of base-config misaligned turns of the same behavior) on v, against the same for
base-config conversations (should be about 0) and for other vectors of the same behavior.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from natsteer.amplify import conv_turn_features
from natsteer.config import DATA_ROOT, REPO_ROOT
from natsteer.monitor import load

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "results" / "monitor_v0"))
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--arms", nargs="*", default=["prompt", "seeds", "full"])
ap.add_argument("--metrics-name", default="metrics_c3e-4.json", help="per-round metrics file to report")
args = ap.parse_args()

L = DATA_ROOT / "loop"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
report = {"detection": defaultdict(dict), "coverage": defaultdict(dict), "steerability": {}}

for arm in args.arms:
    for r in range(20):
        p = L / "arms" / arm / f"r{r}" / args.metrics_name
        if not p.exists():
            break
        m = json.loads(p.read_text())
        report["detection"][arm][r] = {"train_convs": m["train_convs"], "train_pos": m["train_pos"],
                                       **{k: {x: v[x] for x in ("auroc", "auroc_ci", "tpr@1", "tpr@1_ci", "tpr@5", "per_behavior_tpr@1")} for k, v in m["eval"].items()}}
for p in sorted((L / "arms" / "external").glob(f"n*/{args.metrics_name}")):
    m = json.loads(p.read_text())
    report["detection"]["external"][m["size"]] = {"train_convs": m["size"], **{k: {x: v[x] for x in ("auroc", "auroc_ci", "tpr@1", "tpr@1_ci", "tpr@5", "per_behavior_tpr@1")} for k, v in m["eval"].items()}}

# ---------------------------------------------------------------- coverage and diversity
test = load([L / "eval" / "bloom_test.jsonl"], args.layer)
tf = conv_turn_features(test, "pos")
T = np.stack(list(tf.values())) if tf else np.zeros((0, test.T.shape[1]))
train_sets = {}
for arm in args.arms:
    rounds = sorted(int(p.parent.name[1:]) for p in (L / "arms" / arm).glob("r*/candidates.feats.npz"))
    for r in rounds:
        fs = load([L / "arms" / arm / f"r{k}" / "candidates.jsonl" for k in range(r + 1)], args.layer)
        f = conv_turn_features(fs, "pos")
        settings = {json.dumps(fs.convs[c]["meta"].get("setting", {}), sort_keys=True) for c in f}
        train_sets[(arm, r)] = (np.stack(list(f.values())), len(settings))
ext = L / "arms" / "external" / "pool.jsonl"
if ext.with_suffix(".feats.npz").exists():
    fs = load([ext], args.layer)
    for n in (500, 1000, 2000):
        f = conv_turn_features(fs, "pos")
        train_sets[("external", n)] = (np.stack([v for c, v in f.items() if c < n]), 0)
if train_sets and len(T):
    allX = np.concatenate([T] + [v[0] for v in train_sets.values()])
    sc = StandardScaler().fit(allX)
    pca = PCA(50, random_state=0).fit(sc.transform(allX))
    Z = lambda X: pca.transform(sc.transform(X))  # noqa: E731
    Tz = Z(T)
    dT = np.sqrt(((Tz[:, None] - Tz[None]) ** 2).sum(-1))
    np.fill_diagonal(dT, np.inf)
    rho = float(np.median(dT.min(1)))
    for (arm, r), (X, n_set) in train_sets.items():
        Xz = Z(X)
        nn = np.sqrt(((Tz[:, None] - Xz[None]) ** 2).sum(-1)).min(1)
        sub = Xz[np.random.default_rng(0).choice(len(Xz), min(1000, len(Xz)), replace=False)]
        pd = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
        report["coverage"][arm][r] = {"n_pos": len(X), "median_nn_dist": float(np.median(nn)), "covered_share": float(np.mean(nn < rho)),
                                      "mean_pairwise": float(pd[np.triu_indices(len(sub), 1)].mean()), "settings": n_set}
    report["coverage_rho"] = rho

# ---------------------------------------------------------------- steerability of style vectors (full arm)
import torch  # noqa: E402

for r in range(1, 20):
    vp, cp = L / "arms" / "full" / f"vectors_r{r}.pt", L / "arms" / "full" / f"r{r}" / "candidates.jsonl"
    if not (vp.exists() and cp.with_suffix(".feats.npz").exists()):
        continue
    cards = {c["id"]: c for c in torch.load(vp)}
    fs = load([cp], args.layer)
    f = conv_turn_features(fs, "pos")
    by_b = defaultdict(list)
    for c, v in f.items():
        by_b[fs.convs[c]["behavior"]].append((fs.convs[c]["meta"]["config"], v))
    rows = []
    for b, items in by_b.items():
        base = [v for cfg, v in items if cfg == "base"]
        if len(base) < 3:
            continue
        mu = np.mean(base, 0)
        for cid, card in cards.items():
            if card["behavior"] != b:
                continue
            u = card["vec"].numpy() / np.linalg.norm(card["vec"].numpy())
            own = [float((v - mu) @ u) for cfg, v in items if cfg == f"style:{cid}"]
            other = [float((v - mu) @ u) for cfg, v in items if cfg.startswith("style:") and cfg != f"style:{cid}"]
            basep = [float((v - mu) @ u) for v in base]
            if own:
                rows.append({"vector": cid, "n_own": len(own), "own": float(np.mean(own)), "other_styles": float(np.mean(other)) if other else None,
                             "base_sd": float(np.std(basep)), "effect_in_base_sd": float(np.mean(own) / max(np.std(basep), 1e-6))})
    report["steerability"][r] = rows

(out / "report.json").write_text(json.dumps(report, indent=1, default=float))
for arm, d in report["detection"].items():
    for r, v in d.items():
        print(arm, r, v["train_convs"], " ".join(f"{k}: {v[k]['auroc']:.3f}/{v[k]['tpr@1']:.2f}" for k in ("own", "indicators_s", "bloom_val", "bloom_test") if k in v))
for arm, d in report["coverage"].items():
    for r, v in d.items():
        print("coverage", arm, r, {k: round(x, 3) for k, x in v.items()})
for r, rows in report["steerability"].items():
    e = [x["effect_in_base_sd"] for x in rows]
    print("steer r", r, "vectors", len(rows), "mean effect (base sd)", round(float(np.mean(e)), 2) if e else None)

# ---------------------------------------------------------------- figure: Bloom-test detection vs training size
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
style = {"prompt": ("#7f7f7f", "o", "prompt only (no steering, fixed seeds)"), "seeds": ("#1f77b4", "s", "anti-refusal + seed amplification"),
         "full": ("#d62728", "^", "anti-refusal + seeds + style vectors"), "external": ("#2ca02c", "D", "external writer (Claude Sonnet, indicators)")}
for arm, d in report["detection"].items():
    keys = sorted(d, key=lambda k: d[k]["train_convs"])
    xs = [d[k]["train_convs"] for k in keys]
    for ax, met in zip(axes, ("auroc", "tpr@1")):
        ys = [d[k]["bloom_test"][met] for k in keys]
        lo = [d[k]["bloom_test"][f"{met}_ci"][0] for k in keys]
        hi = [d[k]["bloom_test"][f"{met}_ci"][1] for k in keys]
        c, mk, lab = style[arm]
        ax.plot(xs, ys, marker=mk, color=c, label=lab)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12)
axes[0].set_ylabel("AUROC (conversation positives vs. turn negatives)")
axes[1].set_ylabel("detection rate at 1% turn-level false alarms")
for ax in axes:
    ax.set_xlabel("training conversations (loop arms: rounds 0-4)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Monitors on gpt-oss-20b layer 15, tested on Bloom rollouts of GLM-4.7-Flash and Qwen3.6 (test split, 230 misaligned conversations)", fontsize=10)
fig.tight_layout()
fig.savefig(REPO_ROOT / "docs" / "figures" / "loop_rounds.png", dpi=150)
print("wrote docs/figures/loop_rounds.png")
