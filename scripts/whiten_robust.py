"""(CPU; run from the repo root; about 30 min.) A probe that knows normal traffic, over 3 scenario splits.

Normal traffic = windows of gpt-oss-20b's own rollouts from half the on-policy scenarios; test = own rollouts from
the other half, and Bloom test. Arms (writer probe from Claude seeds, layer 15):
  plain      : the writer probe (C = 3e-4)
  negatives  : normal traffic (aligned windows) added to the writer data as negatives (C = 3e-4)
  white      : probe trained on writer data whitened by normal traffic, x -> Sigma^-1/2 (x - mu) (C = 1e-4, chosen on
               Bloom val in whiten_check.py); Sigma from the aligned windows ("clean") or from all windows of the
               fit half, misaligned ones included ("unlabeled": no labels needed)
Output: docs/results/normal_traffic_v0/whiten.json."""
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from natsteer.monitor import evaluate_scores, load

A = "data/loop/arms/"
OUT = Path("docs/results/normal_traffic_v0/whiten.json")
writer = load([A + "seedsrc/claude_s4/r0/candidates.jsonl"], 15)
bt = load(["data/loop/eval/bloom_test.jsonl"], 15)
lab = ~np.isnan(writer.y)
Xw, yw = writer.X[lab].astype(np.float32), writer.y[lab]


def probe(X, y, C):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y)
    return lambda Z: m.decision_function(sc.transform(Z.astype(np.float32)))


def whitened(pool, seed):
    sub = pool[np.random.default_rng(seed).choice(len(pool), min(20000, len(pool)), replace=False)].astype(np.float64)
    mu = sub.mean(0)
    ev, V = np.linalg.eigh(LedoitWolf().fit(sub).covariance_)
    W = (V / np.sqrt(np.clip(ev, 1e-8, None))).astype(np.float32)
    mu = mu.astype(np.float32)
    m = LogisticRegression(C=1e-4, class_weight="balanced", max_iter=2000).fit((Xw - mu) @ W, yw)
    return lambda X: m.decision_function((X.astype(np.float32) - mu) @ W)


def ev(fs, f):
    e = evaluate_scores(fs, f(fs.X), n_boot=0)
    return {k: e[k] for k in ("auroc", "tpr@1", "tpr@5")}


plain = probe(Xw, yw, 3e-4)
out = {"splits": {}}
print(f"{'split':>5} {'arm':>16}  own held-out AUROC / TPR@1 / TPR@5   Bloom test AUROC / TPR@1")
for seed in (0, 1, 2):
    op = load([A + "seedsrc/onpolicy_direct/r0/candidates.jsonl"], 15)
    groups = sorted({c["group"] for c in op.convs})
    fit_g = set(np.random.default_rng(seed).permutation(groups)[: len(groups) // 2])
    in_fit = np.array([op.convs[c]["group"] in fit_g for c in op.w_conv])
    clean, unlab = op.X[in_fit & (op.y == 0)], op.X[in_fit & ~np.isnan(op.y)]
    for c in op.convs:  # test only on the other half
        if c["group"] in fit_g:
            c["turn_labels"] = None
    tested = [c for c in op.convs if c["turn_labels"]]
    s = out["splits"][seed] = {
        "fit_scenarios": len(fit_g), "test_misaligned_convs": sum(c["label"] == 1 for c in tested),
        "test_aligned_turns": sum(t == 0 for c in tested for t in c["turn_labels"]),
        "pool_windows": len(unlab), "pool_misaligned_share": float((op.y[in_fit & ~np.isnan(op.y)] == 1).mean()), "arms": {}}
    arms = {"plain": plain,
            "negatives": probe(np.concatenate([Xw, clean.astype(np.float32)]), np.concatenate([yw, np.zeros(len(clean))]), 3e-4),
            "white_clean": whitened(clean, seed), "white_unlabeled": whitened(unlab, seed)}
    for name, f in arms.items():
        r = s["arms"][name] = {"own": ev(op, f), "bloom": ev(bt, f)}
        print(f"{seed:>5} {name:>16}  {r['own']['auroc']:.3f} / {r['own']['tpr@1']:.2f} / {r['own']['tpr@5']:.2f}"
              f"            {r['bloom']['auroc']:.3f} / {r['bloom']['tpr@1']:.2f}", flush=True)
names = list(out["splits"][0]["arms"])
out["mean"] = {a: {t: {k: float(np.mean([out["splits"][s]["arms"][a][t][k] for s in out["splits"]])) for k in ("auroc", "tpr@1", "tpr@5")}
                   for t in ("own", "bloom")} for a in names}
out["sd"] = {a: {t: {k: float(np.std([out["splits"][s]["arms"][a][t][k] for s in out["splits"]], ddof=1)) for k in ("auroc", "tpr@1", "tpr@5")}
                 for t in ("own", "bloom")} for a in names}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
for a in names:
    m = out["mean"][a]
    print(f"mean {a:>16}  {m['own']['auroc']:.3f} / {m['own']['tpr@1']:.2f} / {m['own']['tpr@5']:.2f}            {m['bloom']['auroc']:.3f} / {m['bloom']['tpr@1']:.2f}")
