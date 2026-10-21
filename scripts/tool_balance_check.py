"""(CPU; run from the repo root; about 10 min.) Can the tool shortcut of the agentic arm be removed at training time?

The agentic writes (arms/agentic_sol) put tool steps into 87% of the misaligned turns but 51% of the aligned branch
turns, so the probe learns "tool calls = misaligned". Variants, each trained on one run's writer data (layer 15):
  plain        the writer data as it is (C 3e-4)
  reweight     window weights so that the share of tool-turn windows is the same in both classes (C 3e-4)
  +own_tool    plus the model's own aligned tool-turn windows from the fit half of the scenarios as negatives
  +own_all     plus all the model's own aligned windows from the fit half as negatives
  white        whitened by the fit half's aligned windows (C 1e-4)
  white+own    whitened, plus all own aligned windows as negatives (C 1e-4)
  +own_tool_unlab  plus all own tool-turn windows of the fit half, misaligned ones included (no labels needed)
  +bloomval_tool   plus aligned tool-turn windows of Bloom val (other models' rollouts: another source than the test)
3 scenario splits (as normal_traffic_v0): test = own rollouts of the other half: all, misaligned vs aligned turns
with tool calls, and vs turns without; also Bloom test. Arms: agentic_sol and branch_sol, 3 runs each.
Output: docs/results/branch_sol_v0/tool_balance.json (--variants and --out to run a subset)."""

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from natsteer.convo import is_call, message_turns
from natsteer.monitor import evaluate_scores, load, turn_table

A = Path("data/loop/arms")
OUT = Path("docs/results/branch_sol_v0/tool_balance.json")
RUNS = {arm: [A / arm / f"s{k}" / "r0" for k in (2, 3, 4)] for arm in ("agentic_sol", "branch_sol")}
VARIANTS = ("plain", "reweight", "+own_tool", "+own_all", "white", "white+own", "+own_tool_unlab", "+bloomval_tool")
ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
VARIANTS = tuple(args.variants)
OUT = Path(args.out)


def window_tool_flags(fs):
    """Per window: does its turn have tool steps (a call or a tool result)?"""
    flag = {}
    for ci, c in enumerate(fs.convs):
        mt = message_turns(c)
        for m, t in zip(c["messages"], mt):
            if m.get("tool") or is_call(m):
                flag[(ci, t)] = True
    return np.array([flag.get((int(c), int(t)), False) for c, t in zip(fs.w_conv, fs.w_turn)])


bt = load(["data/loop/eval/bloom_test.jsonl"], 15)
bv = load(["data/loop/eval/bloom_val.jsonl"], 15)
BV_TOOL = bv.X[(bv.y == 0) & window_tool_flags(bv)].astype(np.float32)
print(f"Bloom val aligned tool-turn windows {len(BV_TOOL)}", flush=True)
splits = []
for seed in (0, 1, 2):
    op = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], 15)
    groups = sorted({c["group"] for c in op.convs})
    fit_g = set(np.random.default_rng(seed).permutation(groups)[: len(groups) // 2])
    in_fit = np.array([op.convs[c]["group"] in fit_g for c in op.w_conv])
    tool_w = window_tool_flags(op)
    clean = in_fit & (op.y == 0)
    sub = op.X[clean][np.random.default_rng(seed).choice(int(clean.sum()), min(20000, int(clean.sum())), replace=False)].astype(np.float64)
    ev, V = np.linalg.eigh(LedoitWolf().fit(sub).covariance_)
    unlab = in_fit & ~np.isnan(op.y) & tool_w
    sp = {"own_all": op.X[clean].astype(np.float32), "own_tool": op.X[clean & tool_w].astype(np.float32),
          "own_tool_unlab": op.X[unlab].astype(np.float32), "unlab_mis_share": float((op.y[unlab] == 1).mean()),
          "mu": sub.mean(0).astype(np.float32), "W": (V / np.sqrt(np.clip(ev, 1e-8, None))).astype(np.float32)}
    for c in op.convs:  # test on the other half only
        if c["group"] in fit_g:
            c["turn_labels"] = None
    tt_tool = {(ci, t): any(m.get("tool") for m, k in zip(c["messages"], message_turns(c)) if k == t)
               for ci, c in enumerate(op.convs) for t, lab in enumerate(c["turn_labels"] or []) if lab == 0}
    sp |= {"fs": op, "tool_turn": tt_tool, "pos": [ci for ci, c in enumerate(op.convs) if c["turn_labels"] and c["label"] == 1]}
    splits.append(sp)
    print(f"split {seed}: own aligned windows {len(sp['own_all'])} (tool turns {len(sp['own_tool'])}); unlabeled tool-turn windows "
          f"{len(sp['own_tool_unlab'])} ({sp['unlab_mis_share']:.1%} misaligned)", flush=True)


def fit(d, variant, k):
    fs = load([d / "candidates.jsonl"], 15)
    lab = ~np.isnan(fs.y)
    X, y = fs.X[lab].astype(np.float32), fs.y[lab]
    sp = splits[k]
    w = None
    if variant == "reweight":  # equal tool-window share in both classes, class totals kept
        tool = window_tool_flags(fs)[lab]
        w = np.ones(len(y))
        target = tool.mean()
        for cls in (0, 1):
            m = y == cls
            share = tool[m].mean()
            if 0 < share < 1:
                w[m & tool] = target / share
                w[m & ~tool] = (1 - target) / (1 - share)
    if variant in ("+own_tool", "+own_all", "white+own", "+own_tool_unlab", "+bloomval_tool"):
        extra = {"+own_tool": sp["own_tool"], "+own_tool_unlab": sp["own_tool_unlab"], "+bloomval_tool": BV_TOOL}.get(variant, sp["own_all"])
        X, y = np.concatenate([X, extra]), np.concatenate([y, np.zeros(len(extra))])
    if variant.startswith("white"):
        Z = (X - sp["mu"]) @ sp["W"]
        m = LogisticRegression(C=1e-4, class_weight="balanced", max_iter=2000).fit(Z, y)
        f = lambda Q: m.decision_function((Q.astype(np.float32) - sp["mu"]) @ sp["W"])  # noqa: E731
    else:
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=3e-4, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
        f = lambda Q: m.decision_function(sc.transform(Q.astype(np.float32)))  # noqa: E731
    op = sp["fs"]
    s = f(op.X)
    e = evaluate_scores(op, s, n_boot=0)
    tt = turn_table(op, s)
    best = {}
    for (c, t), v in tt.items():
        best[c] = max(best.get(c, -np.inf), v)
    pos = np.array([best[c] for c in sp["pos"]])
    tools = []
    for tool in (True, False):
        neg = np.array([tt[kk] for kk, v in sp["tool_turn"].items() if v == tool and kk in tt])
        tools.append(float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])))
    b = evaluate_scores(bt, f(bt.X), n_boot=0)
    return {"own_auroc": e["auroc"], "own_tpr@1": e["tpr@1"], "vs_tool_turns": tools[0], "vs_no_tool_turns": tools[1],
            "bloom_auroc": b["auroc"], "bloom_tpr@1": b["tpr@1"]}


jobs = [(arm, d, v, k) for arm, ds in RUNS.items() for d in ds for v in VARIANTS for k in range(len(splits))]
res = Parallel(n_jobs=min(len(jobs), 64))(delayed(fit)(d, v, k) for arm, d, v, k in jobs)
out = {"runs": [{"arm": arm, "run": str(d), "variant": v, "split": k} | r for (arm, d, v, k), r in zip(jobs, res)], "mean": {}}
KEYS = ("own_auroc", "own_tpr@1", "vs_tool_turns", "vs_no_tool_turns", "bloom_auroc", "bloom_tpr@1")
print(f"\n{'arm':12s} {'variant':16s} " + " ".join(f"{k:>16s}" for k in KEYS))
for arm in RUNS:
    for v in VARIANTS:
        rs = [r for r in out["runs"] if r["arm"] == arm and r["variant"] == v]
        mean = {k: float(np.mean([r[k] for r in rs])) for k in KEYS}
        out["mean"][f"{arm}|{v}"] = mean
        print(f"{arm:12s} {v:16s} " + " ".join(f"{mean[k]:16.3f}" for k in KEYS), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
