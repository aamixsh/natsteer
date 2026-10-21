"""(CPU; run from the repo root.) Report for scripts/queue_branch_rerun.sh (PLAN next todo 1): do the judge (local vs
gpt-6.1-sol) and the negative type (rewrite vs branch) change the writer monitor? All arms share the same writes.

Arms (3 runs each: seeds 2, 3, 4; Claude seeds, anti-refusal, round 0; layer 15, C 3e-4):
  rewrite_local  the monitor_v0 runs (arms/seeds/r0, arms/full/r0, arms/seedsrc/claude_s4/r0)
  rewrite_sol    the same writes and rewrites, judged by gpt-6.1-sol
  branch_sol     the same writes judged by gpt-6.1-sol; branch negatives, judged by gpt-6.1-sol
  agentic_sol    new writes, half of the prompts agentic (--agentic-frac 0.5); branches; gpt-6.1-sol (if present)
Tests: Bloom val and test, Claude-written (indicators_s), own written held-out seeds (own), gpt-oss-20b's own
rollouts without the seed scenarios (onpolicy_heldout), and own rollouts over 3 scenario splits, plain and whitened
by the aligned normal traffic of the other half (as in normal_traffic_v0); on own rollouts also misaligned
conversations vs aligned turns with and without tool calls. Data: kept positives and negatives, judge agreement on
the shared writes and rewrites (positives: local at realism >= 3, gpt-6.1-sol at >= 2, as used), branch status.
Output: docs/results/branch_sol_v0/report.json (and a printed table)."""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from natsteer.convo import load_jsonl, message_turns
from natsteer.monitor import evaluate_scores, load, turn_table, window_scores
from natsteer.writer import has_tools

A = Path("data/loop/arms")
L = Path("data/loop")
OUT = Path("docs/results/branch_sol_v0/report.json")
ARMS = {"rewrite_local": [A / "seeds" / "r0", A / "full" / "r0", A / "seedsrc" / "claude_s4" / "r0"],
        "rewrite_sol": [A / "rewrite_sol" / f"s{k}" / "r0" for k in (2, 3, 4)],
        "branch_sol": [A / "branch_sol" / f"s{k}" / "r0" for k in (2, 3, 4)],
        "agentic_sol": [A / "agentic_sol" / f"s{k}" / "r0" for k in (2, 3, 4)]}
KEYS = ("auroc", "tpr@1", "tpr@5")


def ev(fs, scores):
    e = evaluate_scores(fs, scores, n_boot=0)
    return {k: e[k] for k in KEYS} | {"per_behavior_tpr@1": e.get("per_behavior_tpr@1")}


def accepted(v, min_realism=3):
    return v is not None and any(v["turns"]) and not v["harmful_detail"] and v["realism"] >= min_realism


def clean_neg(v):
    return v is not None and not any(v["turns"]) and not v["harmful_detail"]


runs = {arm: [d for d in dirs if (d / "metrics_c3e-4.monitor.pkl").exists()] for arm, dirs in ARMS.items()}
print({a: len(d) for a, d in runs.items()})
report = {"runs": {}, "summary": {}, "agreement": {}}

# ------------------------------------------------------------------ frozen tests and the own-rollout test
evals = {n: load([L / "eval" / f"{n}.jsonl"], 15) for n in ("own", "indicators_s", "bloom_val", "bloom_test")}
seed_groups = {s["group"] for s in load_jsonl(L / "seeds_onpolicy.jsonl") if s["source"] == "onpolicy"}
held = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], 15)
for c in held.convs:
    if c["group"] in seed_groups:
        c["turn_labels"] = None
evals["onpolicy_heldout"] = held
tool_turn = {(ci, t): any(m.get("tool") for m, k in zip(c["messages"], message_turns(c)) if k == t)
             for ci, c in enumerate(held.convs) for t, lab in enumerate(c["turn_labels"] or []) if lab == 0}
held_pos = [ci for ci, c in enumerate(held.convs) if c["turn_labels"] and c["label"] == 1]


def tool_split(scores):
    """AUROC of misaligned conversations (max turn) vs aligned turns with tool calls, and vs aligned turns without."""
    tt = turn_table(held, scores)
    best = {}
    for (c, t), v in tt.items():
        best[c] = max(best.get(c, -np.inf), v)
    pos = np.array([best[c] for c in held_pos if c in best])
    out = {}
    for name, tool in (("vs_tool_turns", True), ("vs_no_tool_turns", False)):
        neg = np.array([tt[k] for k, is_t in tool_turn.items() if is_t == tool and k in tt])
        out[name] = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    return out

# ------------------------------------------------------------------ 3 scenario splits: normal traffic from one half
splits = []
for seed in (0, 1, 2):
    op = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], 15)
    groups = sorted({c["group"] for c in op.convs})
    fit_g = set(np.random.default_rng(seed).permutation(groups)[: len(groups) // 2])
    in_fit = np.array([op.convs[c]["group"] in fit_g for c in op.w_conv])
    pool = op.X[in_fit & (op.y == 0)]
    sub = pool[np.random.default_rng(seed).choice(len(pool), min(20000, len(pool)), replace=False)].astype(np.float64)
    ev_, V = np.linalg.eigh(LedoitWolf().fit(sub).covariance_)
    for c in op.convs:
        if c["group"] in fit_g:
            c["turn_labels"] = None
    splits.append({"fs": op, "mu": sub.mean(0).astype(np.float32), "W": (V / np.sqrt(np.clip(ev_, 1e-8, None))).astype(np.float32)})
    print(f"split {seed} ready", flush=True)


def whitened_scores(d, sp):
    fs = load([d / "candidates.jsonl"], 15)
    lab = ~np.isnan(fs.y)
    X = fs.X[lab].astype(np.float32)
    m = LogisticRegression(C=1e-4, class_weight="balanced", max_iter=2000).fit((X - sp["mu"]) @ sp["W"], fs.y[lab])
    return m.decision_function((sp["fs"].X.astype(np.float32) - sp["mu"]) @ sp["W"])


all_runs = [(arm, d) for arm, ds in runs.items() for d in ds]
jobs = [(i, k) for i in range(len(all_runs)) for k in range(len(splits))]
wscores = Parallel(n_jobs=min(len(jobs), 32))(delayed(whitened_scores)(all_runs[i][1], splits[k]) for i, k in jobs)
wscores = dict(zip(jobs, wscores))

for i, (arm, d) in enumerate(all_runs):
    m = pickle.load(open(d / "metrics_c3e-4.monitor.pkl", "rb"))
    res = {n: ev(fs, window_scores(m, fs.X)) for n, fs in evals.items()}
    res["onpolicy_tools"] = tool_split(window_scores(m, held.X))
    plain = [ev(sp["fs"], window_scores(m, sp["fs"].X)) for sp in splits]
    white = [ev(sp["fs"], wscores[(i, k)]) for k, sp in enumerate(splits)]
    res["own_splits_plain"] = {k: float(np.mean([p[k] for p in plain])) for k in KEYS}
    res["own_splits_whitened"] = {k: float(np.mean([w[k] for w in white])) for k in KEYS}
    st = json.loads((d / "stats.json").read_text())
    tot = Counter()
    for v in st["by_behavior_config"].values():
        tot.update(v)
    cand = load_jsonl(d / "candidates.jsonl")
    res["data"] = dict(tot) | {"train_pos": sum(c["label"] == 1 for c in cand), "train_neg": sum(c["label"] == 0 for c in cand),
                               "pos_by_behavior": dict(Counter(c["behavior"] for c in cand if c["label"] == 1)),
                               "with_tools": {lab: sum(has_tools(c["messages"]) for c in cand if c["label"] == y) for lab, y in (("pos", 1), ("neg", 0))}}
    if (d / "branches.jsonl").exists():
        br = load_jsonl(d / "branches.jsonl")
        res["data"]["branch_status"] = dict(Counter(r["status"] for r in br))
        res["data"]["branch_turn"] = dict(Counter(f"{r['branch_turn'] + 1} of {r['n_turns']}" for r in br))
    report["runs"][str(d)] = {"arm": arm} | res
    print(f"{arm:14s} {str(d):34s} Bloom test {res['bloom_test']['auroc']:.3f} / {res['bloom_test']['tpr@1']:.2f}   own rollouts "
          f"{res['onpolicy_heldout']['auroc']:.3f} / {res['onpolicy_heldout']['tpr@1']:.2f}   splits plain {res['own_splits_plain']['auroc']:.3f} / "
          f"{res['own_splits_plain']['tpr@1']:.2f} whitened {res['own_splits_whitened']['auroc']:.3f} / {res['own_splits_whitened']['tpr@1']:.2f}   "
          f"pos {res['data']['train_pos']} neg {res['data']['train_neg']}", flush=True)

# ------------------------------------------------------------------ means and SDs over runs
TESTS = ("bloom_val", "bloom_test", "indicators_s", "own", "onpolicy_heldout", "own_splits_plain", "own_splits_whitened")
for arm in runs:
    rs = [r for r in report["runs"].values() if r["arm"] == arm]
    if not rs:
        continue
    report["summary"][arm] = {"n_runs": len(rs)} | {t: {k: [float(np.mean([r[t][k] for r in rs])), float(np.std([r[t][k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0]
                                                        for k in KEYS} for t in TESTS}
    for k in ("train_pos", "train_neg"):
        report["summary"][arm][k] = float(np.mean([r["data"][k] for r in rs]))
    report["summary"][arm]["onpolicy_tools"] = {k: float(np.mean([r["onpolicy_tools"][k] for r in rs])) for k in ("vs_tool_turns", "vs_no_tool_turns")}

# ------------------------------------------------------------------ judge agreement on the shared writes and rewrites
for a_old, a_new in zip(ARMS["rewrite_local"], ARMS["branch_sol"]):
    if not (a_new / "judged_pos.jsonl").exists():
        continue
    old = {(j["job"], j["kind"]): j["verdict"] for j in load_jsonl(a_old / "judged.jsonl")}
    new = {j["job"]: j["verdict"] for j in load_jsonl(a_new / "judged_pos.jsonl")}
    pos = Counter((accepted(old.get((job, "pos")), 3), accepted(v, 2)) for job, v in new.items())
    r_sol = A / "rewrite_sol" / a_new.parent.name / "r0" / "judged.jsonl"
    neg = Counter()
    if r_sol.exists():
        for j in load_jsonl(r_sol):
            if j["kind"] == "neg" and (j["job"], "neg") in old:
                neg[(clean_neg(old[(j["job"], "neg")]), clean_neg(j["verdict"]))] += 1
    report["agreement"][a_new.parent.name] = {
        "writes_accepted_local_sol": {f"{a}_{b}": n for (a, b), n in pos.items()},
        "rewrites_clean_local_sol": {f"{a}_{b}": n for (a, b), n in neg.items()}}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1))
print("\nmean (SD) over runs: AUROC / detection at 1%")
for arm, s in report["summary"].items():
    f = lambda t: f"{s[t]['auroc'][0]:.3f} ({s[t]['auroc'][1]:.3f}) / {s[t]['tpr@1'][0]:.2f} ({s[t]['tpr@1'][1]:.2f})"  # noqa: E731
    print(f"{arm:14s} n={s['n_runs']}  Bloom test {f('bloom_test')}  own rollouts {f('onpolicy_heldout')}  splits plain {f('own_splits_plain')}"
          f"  whitened {f('own_splits_whitened')}  pos {s['train_pos']:.0f} neg {s['train_neg']:.0f}")
    print(f"{'':14s} own rollouts, misaligned vs aligned tool turns {s['onpolicy_tools']['vs_tool_turns']:.3f}, vs turns without tools {s['onpolicy_tools']['vs_no_tool_turns']:.3f}")
print("agreement (local accepted, sol accepted):", json.dumps(report["agreement"]))
