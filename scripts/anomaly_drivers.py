"""(CPU; run from the repo root.) What drives the anomaly score of a turn, and how big is the misalignment direction relative to normal variance?
Output: docs/results/normal_traffic_v0/anomaly_drivers.json."""
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from natsteer.convo import turn_of_messages
from natsteer.monitor import load

A = "data/loop/arms/"
mon = pickle.load(open(A + "seedsrc/claude_s4/r0/metrics_c3e-4.monitor.pkl", "rb"))
sc, lr = mon.steps[0][1], mon.steps[1][1]
w = lr.coef_[0] / sc.scale_
u = w / np.linalg.norm(w)  # probe direction in raw feature space


def turns(fs, keep):
    nwin = Counter(zip(fs.w_conv.tolist(), fs.w_turn.tolist()))
    rows = []
    for i, (c, t) in enumerate(zip(fs.t_conv.tolist(), fs.t_turn.tolist())):
        if c not in keep:
            continue
        tl = fs.convs[c].get("turn_labels") or []
        if t >= len(tl) or tl[t] is None:
            continue
        msgs = fs.convs[c]["messages"]
        tom = turn_of_messages(msgs)
        tool = any(m["role"] == "user" and m.get("tool") for m, k in zip(msgs, tom) if k == t)
        rows.append(dict(x=fs.T[i].astype(np.float64), y=int(tl[t]), b=fs.convs[c]["behavior"], t=t, n=nwin[(c, t)], tool=tool))
    return rows


def report(name, fit_rows, test_rows):
    F = np.stack([r["x"] for r in fit_rows])
    pca = PCA(64, random_state=0).fit(F)
    cov = LedoitWolf().fit(pca.transform(F))
    X = np.stack([r["x"] for r in test_rows])
    a = cov.mahalanobis(pca.transform(X))
    y = np.array([r["y"] for r in test_rows])
    n = np.array([r["n"] for r in test_rows])
    t = np.array([r["t"] for r in test_rows])
    tool = np.array([r["tool"] for r in test_rows])
    al = y == 0
    print(f"\n== {name}: {int(y.sum())} misaligned, {int(al.sum())} aligned turns")
    print(f"  anomaly vs turn length (windows), aligned turns: Spearman {spearmanr(a[al], n[al])[0]:+.2f}")
    print(f"  anomaly vs turn index, aligned turns:            Spearman {spearmanr(a[al], t[al])[0]:+.2f}")
    if tool.any() and (~tool).any():
        print(f"  anomaly AUROC, aligned tool turns vs aligned no-tool turns: {roc_auc_score(tool[al], a[al]):.2f}  (tool turns {int(tool[al].sum())})")
        print(f"  share of tool turns: misaligned {tool[y == 1].mean():.2f}, aligned {tool[al].mean():.2f}")
    print(f"  median length (windows): misaligned {np.median(n[y == 1]):.0f}, aligned {np.median(n[al]):.0f}")
    # top-decile anomalies among aligned turns: what are they?
    hi = al & (a >= np.quantile(a[al], 0.9))
    print(f"  top-10% anomalous aligned turns: median length {np.median(n[hi]):.0f}, tool share {tool[hi].mean():.2f}, mean turn index {t[hi].mean():.2f}")
    # size of the misalignment direction
    Fc = F - F.mean(0)
    var_u = np.var(Fc @ u)
    tot = np.var(Fc, 0).sum()
    ev = pca.explained_variance_
    print(f"  normal variance along probe direction: {var_u / tot:.4%} of total (a random direction: {1 / F.shape[1]:.4%}); "
          f"top PC carries {ev[0] / tot:.1%}, top 64 PCs {ev.sum() / tot:.1%}")
    print(f"  share of probe direction inside the top-64 PC space: {np.linalg.norm(pca.components_ @ u) ** 2:.1%}")
    d = (X[y == 1] @ u).mean() - (X[al] @ u).mean()
    pc1 = abs(((X[y == 1] - X[al].mean(0)) @ pca.components_[0]).mean()) / np.sqrt(ev[0])
    print(f"  misaligned-minus-aligned along probe direction: {d / np.sqrt(var_u):.2f} normal SD; along top PC: {pc1:.2f} SD")
    hi_t = tool[hi].mean() if tool.any() else None
    return {"misaligned_turns": int(y.sum()), "aligned_turns": int(al.sum()),
            "spearman_anomaly_length": float(spearmanr(a[al], n[al])[0]), "spearman_anomaly_turn_index": float(spearmanr(a[al], t[al])[0]),
            "anomaly_auroc_tool_vs_no_tool": float(roc_auc_score(tool[al], a[al])) if tool[al].any() and (~tool[al]).any() else None,
            "median_windows": {"misaligned": float(np.median(n[y == 1])), "aligned": float(np.median(n[al])), "top10_anomalous_aligned": float(np.median(n[hi]))},
            "top10_anomalous_aligned_tool_share": None if hi_t is None else float(hi_t),
            "probe_dir_var_share": float(var_u / tot), "random_dir_var_share": 1 / F.shape[1], "top_pc_var_share": float(ev[0] / tot),
            "top64_var_share": float(ev.sum() / tot), "probe_dir_in_top64": float(np.linalg.norm(pca.components_ @ u) ** 2),
            "shift_along_probe_sd": float(d / np.sqrt(var_u)), "shift_along_top_pc_sd": float(pc1)}


op = load([A + "seedsrc/onpolicy_direct/r0/candidates.jsonl"], 15)
groups = sorted({c["group"] for c in op.convs})
fit_g = set(np.random.default_rng(0).permutation(groups)[: len(groups) // 2])
fit_c = {i for i, c in enumerate(op.convs) if c["group"] in fit_g}
rows_fit = [r for r in turns(op, fit_c) if r["y"] == 0]
out = {}
out["own"] = report("own rollouts (held-out scenarios)", rows_fit, turns(op, set(range(len(op.convs))) - fit_c))

bv = load(["data/loop/eval/bloom_val.jsonl"], 15)
bt = load(["data/loop/eval/bloom_test.jsonl"], 15)
out["bloom"] = report("Bloom test (normal = Bloom val aligned)", [r for r in turns(bv, set(range(len(bv.convs)))) if r["y"] == 0],
       turns(bt, set(range(len(bt.convs)))))
Path("docs/results/normal_traffic_v0/anomaly_drivers.json").write_text(json.dumps(out, indent=1))
