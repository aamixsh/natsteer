"""(CPU; run from the repo root.) Are misaligned turns anomalous relative to normal (aligned) traffic? Does an anomaly gate in front of the
probe help at low false-alarm rates? Turn-level, layer 15, existing features only (CPU).
Output: docs/results/normal_traffic_v0/anomaly_gate.json (anomaly AUROC, gate table, anomaly-score histograms)."""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from natsteer.monitor import load, turn_table, window_scores

A = "data/loop/arms/"
OUT = Path("docs/results/normal_traffic_v0/anomaly_gate.json")


def turn_feats(fs):
    """{(conv, turn): (feature, label)} for labeled turns."""
    out = {}
    for i, (c, t) in enumerate(zip(fs.t_conv, fs.t_turn)):
        tl = fs.convs[c].get("turn_labels") or []
        if t < len(tl) and tl[t] is not None:
            out[(int(c), int(t))] = (fs.T[i].astype(np.float32), int(tl[t]))
    return out


class Normal:
    def __init__(self, X, k=64):
        self.pca = PCA(k, random_state=0).fit(X)
        Z = self.pca.transform(X)
        self.cov = LedoitWolf().fit(Z)

    def score(self, X):
        return self.cov.mahalanobis(self.pca.transform(X))


def cascade_eval(name, fs, keep_conv, normal, monitor, fprs=(0.01, 0.001)):
    tf = turn_feats(fs)
    keys = [k for k in tf if k[0] in keep_conv]
    X = np.stack([tf[k][0] for k in keys])
    y = np.array([tf[k][1] for k in keys])
    an = dict(zip(keys, normal.score(X)))
    tt = turn_table(fs, window_scores(monitor, fs.X))
    a_all = np.array([an[k] for k in keys])
    res = {"misaligned_turns": int(y.sum()), "aligned_turns": int((y == 0).sum()), "anomaly_auroc": float(roc_auc_score(y, a_all)),
           "anomaly_auroc_by_behavior": {}, "gate": {}}
    print(f"\n{name}: {res['misaligned_turns']} misaligned turns, {res['aligned_turns']} aligned turns")
    print(f"  anomaly AUROC, misaligned vs aligned turns: {res['anomaly_auroc']:.3f}")
    by_b = defaultdict(list)
    for k, l in zip(keys, y):
        by_b[fs.convs[k[0]]["behavior"]].append((an[k], l))
    for b, v in sorted(by_b.items()):
        a, l = np.array(v).T
        if 0 < l.sum() < len(l):
            res["anomaly_auroc_by_behavior"][b] = {"auroc": float(roc_auc_score(l, a)), "misaligned": int(l.sum())}
            print(f"    {b:17s} anomaly AUROC {res['anomaly_auroc_by_behavior'][b]['auroc']:.3f}  (misaligned {int(l.sum())})")
    # histogram of log anomaly scores (shared bins) for the figure
    la = np.log(a_all)
    bins = np.linspace(np.quantile(la, 0.005), np.quantile(la, 0.995), 41)
    res["hist"] = {"bins": bins.tolist(), "aligned": np.histogram(la[y == 0], bins)[0].tolist(), "misaligned": np.histogram(la[y == 1], bins)[0].tolist()}
    # indicators convention: positive = misaligned conversation (max over its turns), negative = aligned turn
    neg_keys = [k for k, l in zip(keys, y) if l == 0]
    pos_conv = sorted({k[0] for k, l in zip(keys, y) if l == 1})
    neg_an = np.array([an[k] for k in neg_keys])
    print(f"  {'anomaly gate passes':>22s} " + "  ".join(f"TPR@{f:g}" for f in fprs))
    for q in (1.0, 0.5, 0.2, 0.1):
        a_thr = np.quantile(neg_an, 1 - q) if q < 1 else -np.inf
        gated = lambda k: tt.get(k, -np.inf) if an[k] >= a_thr else -np.inf  # noqa: E731
        neg_s = np.array([gated(k) for k in neg_keys])
        pos_s = np.array([max(gated(k) for k in keys if k[0] == c) for c in pos_conv])
        row = []
        for f in fprs:
            finite = neg_s[np.isfinite(neg_s)]
            # total false alarms = share of aligned turns that pass the gate AND exceed the probe threshold
            if q <= f:
                row.append(float(np.mean(np.isfinite(pos_s))))  # the gate alone is within budget
                continue
            thr = np.quantile(finite, 1 - f / q, method="higher")
            row.append(float(np.mean(pos_s > thr)))
        res["gate"][str(q)] = {f"tpr@{f:g}": r for f, r in zip(fprs, row)}
        print(f"  {f'{q:.0%} of aligned turns':>22s} " + "  ".join(f"{r:7.2f}" for r in row))
    return res


mon = pickle.load(open(A + "seedsrc/claude_s4/r0/metrics_c3e-4.monitor.pkl", "rb"))
report = {}

# 1) the model's own rollouts: normal model from aligned turns of half the scenarios, test on the other half
op = load([A + "seedsrc/onpolicy_direct/r0/candidates.jsonl"], 15)
groups = sorted({c["group"] for c in op.convs})
rng = np.random.default_rng(0)
fit_g = set(rng.permutation(groups)[: len(groups) // 2])
tf = turn_feats(op)
fitX = np.stack([v[0] for k, v in tf.items() if v[1] == 0 and op.convs[k[0]]["group"] in fit_g])
normal_op = Normal(fitX)
report["own"] = cascade_eval("own rollouts (held-out scenarios), normal = own aligned turns", op,
                             {i for i, c in enumerate(op.convs) if c["group"] not in fit_g}, normal_op, mon)

# 2) Bloom: normal model from Bloom val aligned turns, test on Bloom test
bv = load(["data/loop/eval/bloom_val.jsonl"], 15)
bt = load(["data/loop/eval/bloom_test.jsonl"], 15)
tfv = turn_feats(bv)
normal_b = Normal(np.stack([v[0] for v in tfv.values() if v[1] == 0]))
report["bloom"] = cascade_eval("Bloom test, normal = Bloom val aligned turns", bt, set(range(len(bt.convs))), normal_b, mon)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1))
