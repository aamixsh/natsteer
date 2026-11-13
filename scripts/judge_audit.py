"""(CPU; run from the repo root.) Judge audit (PLAN next todo 1): do the judges agree, and do the monitor results
hold under other judges' labels? Reads data/judges/<judge>/<set>.jsonl from scripts/judge_ensemble.py.

  judge_audit.py [--judges qwen3.6-35b-a3b qwen3.8-27b gpt-oss-20b] [--no-monitors]

Label sources: sol (gpt-6.1-sol, 1 sample, the labels in use), each local judge (per-turn majority of its samples)
and maj (per-turn majority of sol and the first two local judges, items all three parsed).
(a) Agreement per set (onpolicy, writes, branches) and behavior: positive rate (a conversation is positive when a
    turn is 1), Cohen's kappa per judge pair at conversation and turn level, and self-consistency (share of items
    whose samples all give the same conversation label).
(b) The branch_sol_v0 monitors (rewrite_local, rewrite_sol, branch_sol, agentic_sol; 3 runs each) scored on the own
    rollouts without the seed scenarios (onpolicy_heldout) with each label source: if the arm order holds under the
    Qwen labels, the result is not only agreement with gpt-6.1-sol.
Output: docs/results/judge_audit_v0/report.json (and a printed table).
"""

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from natsteer.convo import load_jsonl

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--judges", nargs="*", default=["qwen3.6-35b-a3b", "qwen3.8-27b", "gpt-oss-20b"])
ap.add_argument("--dir", default="data/judges")
ap.add_argument("--no-monitors", action="store_true")
ap.add_argument("--out", default="docs/results/judge_audit_v0/report.json")
args = ap.parse_args()

A = Path("data/loop/arms")
SETS = ("onpolicy", "writes", "branches")

# ------------------------------------------------------------------ labels: {set: {source: {id: turns}}}, behavior per id
labels = {s: defaultdict(dict) for s in SETS}
behavior, consistency = {}, defaultdict(lambda: defaultdict(list))
for c in load_jsonl(A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"):
    labels["onpolicy"]["sol"][c["id"]] = c["turn_labels"]
    behavior[c["id"]] = c["behavior"]
for k in (2, 3, 4):
    d = A / "branch_sol" / f"s{k}" / "r0"
    for w in load_jsonl(d / "writes.jsonl"):
        behavior[f"s{k}|{w['job']}"] = behavior[f"s{k}|{w['job']}:neg"] = w["behavior"]
    for j in load_jsonl(d / "judged_pos.jsonl"):
        if j["verdict"] is not None:
            labels["writes"]["sol"][f"s{k}|{j['job']}"] = j["verdict"]["turns"]
    for j in load_jsonl(d / "judged_neg.jsonl"):
        if j["verdict"] is not None:
            labels["branches"]["sol"][f"s{k}|{j['job']}:neg"] = j["verdict"]["turns"]
judges = [j for j in args.judges if (Path(args.dir) / j).exists()]
for j in judges:
    for s in SETS:
        p = Path(args.dir) / j / f"{s}.jsonl"
        if not p.exists():
            continue
        for r in load_jsonl(p):
            behavior[r["id"]] = r["behavior"]
            if r["turns"] is not None:
                labels[s][j][r["id"]] = r["turns"]
                conv = [int(any(v["turns"])) for v in r["samples"] if v is not None]
                consistency[s][j].append(len(set(conv)) == 1)
for s in SETS:
    trio = [x for x in ["sol"] + judges[:2] if x in labels[s]]
    if len(trio) == 3:
        for i in set.intersection(*(set(labels[s][x]) for x in trio)):
            ts = [labels[s][x][i] for x in trio]
            if len({len(t) for t in ts}) == 1:
                labels[s]["maj"][i] = [int(sum(col) >= 2) for col in zip(*ts)]
sources = ["sol"] + judges + ["maj"]


def kappa(a, b):
    return float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else float("nan")


# ------------------------------------------------------------------ (a) agreement
report = {"judges": judges, "agreement": {}, "monitors": {}}
for s in SETS:
    srcs = [x for x in sources if labels[s].get(x)]
    rs = {"n": {x: len(labels[s][x]) for x in srcs}, "self_consistency": {j: float(np.mean(v)) for j, v in consistency[s].items()}}
    for b in [None] + sorted({behavior[i] for i in labels[s]["sol"]}):
        key = b or "all"
        rate = {x: float(np.mean([any(t) for i, t in labels[s][x].items() if b in (None, behavior[i])])) for x in srcs}
        pairs = {}
        for x, y in itertools.combinations(srcs, 2):
            ids = [i for i in labels[s][x] if i in labels[s][y] and b in (None, behavior[i]) and len(labels[s][x][i]) == len(labels[s][y][i])]
            if not ids:
                continue
            ca, cb = [int(any(labels[s][x][i])) for i in ids], [int(any(labels[s][y][i])) for i in ids]
            ta = [v for i in ids for v in labels[s][x][i]]
            tb = [v for i in ids for v in labels[s][y][i]]
            pairs[f"{x}~{y}"] = {"n": len(ids), "conv_agree": float(np.mean(np.array(ca) == np.array(cb))), "conv_kappa": kappa(ca, cb),
                                 "turn_agree": float(np.mean(np.array(ta) == np.array(tb))), "turn_kappa": kappa(ta, tb)}
        rs[key] = {"positive_rate": rate, "pairs": pairs}
    report["agreement"][s] = rs

for s in SETS:
    if s not in report["agreement"] or len(report["agreement"][s]["n"]) < 2:
        continue
    r = report["agreement"][s]
    print(f"\n== {s}: items {r['n']}, self-consistency {({j: round(v, 2) for j, v in r['self_consistency'].items()})}")
    for key in [k for k in r if k not in ("n", "self_consistency")]:
        rate = "  ".join(f"{x} {v:.2f}" for x, v in r[key]["positive_rate"].items())
        kap = "  ".join(f"{p} {v['conv_kappa']:.2f}/{v['turn_kappa']:.2f}" for p, v in r[key]["pairs"].items() if "maj" not in p)
        print(f"  {key:18s} positive rate: {rate}\n  {'':18s} kappa conv/turn: {kap}")

# ------------------------------------------------------------------ (b) monitors on own rollouts, per label source
if not args.no_monitors:
    from natsteer.monitor import evaluate_scores, load, window_scores

    ARMS = {"rewrite_local": [A / "seeds" / "r0", A / "full" / "r0", A / "seedsrc" / "claude_s4" / "r0"],
            "rewrite_sol": [A / "rewrite_sol" / f"s{k}" / "r0" for k in (2, 3, 4)],
            "branch_sol": [A / "branch_sol" / f"s{k}" / "r0" for k in (2, 3, 4)],
            "agentic_sol": [A / "agentic_sol" / f"s{k}" / "r0" for k in (2, 3, 4)]}
    seed_groups = {s["group"] for s in load_jsonl(Path("data/loop/seeds_onpolicy.jsonl")) if s["source"] == "onpolicy"}
    held = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], 15)
    runs = [(arm, d) for arm, ds in ARMS.items() for d in ds if (d / "metrics_c3e-4.monitor.pkl").exists()]
    scores = {d: window_scores(pickle.load(open(d / "metrics_c3e-4.monitor.pkl", "rb")), held.X) for _, d in runs}
    for src in [x for x in sources if labels["onpolicy"].get(x)]:
        lab = labels["onpolicy"][src]
        for c in held.convs:
            c["turn_labels"] = None if c["group"] in seed_groups else lab.get(c["id"])  # evaluate_scores reads these
        per_arm = defaultdict(list)
        for arm, d in runs:
            e = evaluate_scores(held, scores[d], n_boot=0)
            per_arm[arm].append(e)
        report["monitors"][src] = {arm: {"n_runs": len(es), "n_pos": es[0]["n_pos"], "n_neg_turns": es[0]["n_neg_turns"]}
                                   | {k: [float(np.mean([e[k] for e in es])), float(np.std([e[k] for e in es], ddof=1)) if len(es) > 1 else 0.0]
                                      for k in ("auroc", "tpr@1", "tpr@5")} for arm, es in per_arm.items()}
    print("\nmonitors on own rollouts (held-out scenarios), mean (SD) over runs: AUROC / detection at 1%")
    for src, arms in report["monitors"].items():
        n = next(iter(arms.values()))
        print(f"  labels {src:16s} (pos {n['n_pos']}, aligned turns {n['n_neg_turns']}): " + "  ".join(
            f"{arm} {v['auroc'][0]:.3f} ({v['auroc'][1]:.3f}) / {v['tpr@1'][0]:.2f}" for arm, v in arms.items()))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=1))
print(f"\nwrote {args.out}")
