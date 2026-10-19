"""Report: does the seed source matter? Claude-written seeds vs on-policy seeds vs training on on-policy rollouts.

  seedsrc_report.py [--out docs/results/onpolicy_v0]

Arms (all round 0, anti-refusal writer, 500 prompts, C = 3e-4, layer 15):
  claude    3 replicates from data/loop/seeds_r0.jsonl (seeds/r0 seed 2, full/r0 seed 3, seedsrc/claude_s4 seed 4)
  onpolicy  3 replicates from data/loop/seeds_onpolicy.jsonl (seedsrc/onpolicy_s2..s4); sabotage and sandbagging
            keep Claude seeds (no on-policy positives)
  direct    a monitor trained on the judged on-policy rollouts themselves (seedsrc/onpolicy_direct)
  prompt    reference: prompt-only round 0 (no anti-refusal; prompt/r0)
Test sets: the frozen loop sets (own, indicators_s, bloom_val, bloom_test) and the on-policy rollouts with the
seed scenarios removed (onpolicy_heldout). The direct arm is scored on the on-policy rollouts out of fold
(5 folds by scenario). Also summarizes the pilot (misbehavior rates) and the generation yield per arm.
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from natsteer.config import DATA_ROOT, REPO_ROOT
from natsteer.convo import load_jsonl
from natsteer.monitor import evaluate, evaluate_scores, load, train, window_scores

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "results" / "onpolicy_v0"))
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--C", type=float, default=3e-4)
args = ap.parse_args()

L = DATA_ROOT / "loop"
A = L / "arms"
PILOT = DATA_ROOT / "onpolicy" / "pilot"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
RUNS = {"claude": [A / "seeds" / "r0", A / "full" / "r0", A / "seedsrc" / "claude_s4" / "r0"],
        "onpolicy": [A / "seedsrc" / f"onpolicy_s{k}" / "r0" for k in (2, 3, 4)],
        "direct": [A / "seedsrc" / "onpolicy_direct" / "r0"],
        "prompt": [A / "prompt" / "r0"]}
EVALS = ("own", "indicators_s", "bloom_val", "bloom_test", "onpolicy_heldout")
KEEP = ("auroc", "auroc_ci", "tpr@1", "tpr@1_ci", "tpr@5", "n_pos", "n_neg_turns", "per_behavior_tpr@1")

# ------------------------------------------------------------------ on-policy test set (seed scenarios removed)
op = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], args.layer)
seed_groups = {s["group"] for s in load_jsonl(L / "seeds_onpolicy.jsonl") if s["source"] == "onpolicy"}
held = load([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"], args.layer)
for c in held.convs:
    if c["group"] in seed_groups:
        c["turn_labels"] = None  # excluded from scoring
evals = {n: load([L / "eval" / f"{n}.jsonl"], args.layer) for n in EVALS[:4]}
evals["onpolicy_heldout"] = held

report = {"pilot": json.loads((PILOT / "report.json").read_text()), "seed_groups_removed": sorted(seed_groups), "runs": {}, "summary": {}}
for arm, dirs in RUNS.items():
    for d in dirs:
        mp = d / "metrics_c3e-4.monitor.pkl"
        if not mp.exists():
            print("missing", mp)
            continue
        m = pickle.load(open(mp, "rb"))
        res = {}
        for n, fs in evals.items():
            if arm == "direct" and n == "onpolicy_heldout":
                continue
            res[n] = {k: v for k, v in evaluate(fs, m).items() if k in KEEP}
        if arm == "direct":  # out of fold on all on-policy rollouts, folds by scenario
            groups = sorted({c["group"] for c in op.convs})
            fold = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(groups))}
            wf = np.array([fold[op.convs[c]["group"]] for c in op.w_conv])
            ws = np.zeros(len(op.X))
            lab = ~np.isnan(op.y)
            for k in range(5):
                mk = train(op.X[lab & (wf != k)], op.y[lab & (wf != k)], args.C)
                ws[wf == k] = window_scores(mk, op.X[wf == k])
            res["onpolicy_heldout"] = {k: v for k, v in evaluate_scores(op, ws).items() if k in KEEP} | {"note": "out of fold, all scenarios"}
        cand = load_jsonl(d / "candidates.jsonl")
        st = json.loads((d / "stats.json").read_text()) if (d / "stats.json").exists() else None
        gen = None
        if st:
            tot = Counter()
            for v in st["by_behavior_config"].values():
                tot.update(v)
            gen = {"prompts": tot["prompts"], "write_ok": tot["write_ok"], "accepted_pos": tot["accepted_pos"], "accepted_neg": tot["accepted_neg"],
                   "pos_by_behavior": dict(Counter(c["behavior"] for c in cand if c["label"] == 1))}
        report["runs"][str(d.relative_to(A))] = {"arm": arm, "train_convs": len(cand), "train_pos": sum(c["label"] == 1 for c in cand), "generation": gen, "eval": res}
        print(arm, d.relative_to(A), " ".join(f"{n}: {r['auroc']:.3f}/{r['tpr@1']:.2f}" for n, r in res.items()), flush=True)

for arm in RUNS:
    rs = [r for r in report["runs"].values() if r["arm"] == arm]
    if not rs:
        continue
    summ = {}
    for n in EVALS:
        vals = [r["eval"][n] for r in rs if n in r["eval"]]
        if not vals:
            continue
        summ[n] = {"n_runs": len(vals), "auroc_mean": float(np.mean([v["auroc"] for v in vals])), "auroc_sd": float(np.std([v["auroc"] for v in vals])),
                   "tpr@1_mean": float(np.mean([v["tpr@1"] for v in vals])), "tpr@1_sd": float(np.std([v["tpr@1"] for v in vals])),
                   "per_behavior_tpr@1_mean": {b: float(np.mean([v["per_behavior_tpr@1"][b]["tpr"] for v in vals if b in v["per_behavior_tpr@1"]]))
                                               for b in sorted({b for v in vals for b in v["per_behavior_tpr@1"]})}}
    report["summary"][arm] = summ
(out / "report.json").write_text(json.dumps(report, indent=1, default=float))

print("\nmean (sd) over runs: AUROC / detection at 1% false alarms")
for arm, summ in report["summary"].items():
    print(f"{arm:9s}", "  ".join(f"{n}: {s['auroc_mean']:.3f}({s['auroc_sd']:.3f}) / {s['tpr@1_mean']:.2f}({s['tpr@1_sd']:.2f})" for n, s in summ.items()))
for arm, summ in report["summary"].items():
    if "bloom_test" in summ:
        print(arm, "bloom_test per behavior", {b: round(v, 2) for b, v in summ["bloom_test"]["per_behavior_tpr@1_mean"].items()})

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cells = report["pilot"]["by_cell"]
behs = ["sycophancy", "self_preservation", "deception", "sabotage", "sandbagging"]
fig, ax = plt.subplots(figsize=(9, 3.4))
for i, (cfg, col, lab) in enumerate((("none", "#7f7f7f", "no steering"), ("anti_refusal", "#1f77b4", "anti-refusal"))):
    rates, ns = [], []
    for b in behs:
        mis = sum(v.get("misaligned", 0) for k, v in cells.items() if k.startswith(b + "|") and k.endswith("|" + cfg))
        n = sum(v.get("judged", 0) for k, v in cells.items() if k.startswith(b + "|") and k.endswith("|" + cfg))
        rates.append(mis / max(n, 1))
        ns.append(f"{mis}/{n}")
    bars = ax.bar(np.arange(len(behs)) + (i - 0.5) * 0.38, rates, 0.38, color=col, label=lab)
    for bar, t in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, t, ha="center", fontsize=8)
ax.set_xticks(range(len(behs)), [b.replace("_", "-") for b in behs])
ax.set_ylabel("rollouts judged misaligned")
ax.set_ylim(0, 1.1)
ax.legend(frameon=False, fontsize=9)
ax.set_title("gpt-oss-20b in 100 pressure scenarios (4 samples each, 4 turns)", fontsize=10)
fig.tight_layout()
fig.savefig(REPO_ROOT / "docs" / "figures" / "onpolicy_rates.png", dpi=130)

S = report["summary"]
arms = [a for a in ("prompt", "claude", "onpolicy", "direct") if a in S]
labels = {"prompt": "prompt only\n(Claude seeds, 1 run)", "claude": "Claude seeds\n(3 runs)", "onpolicy": "on-policy seeds\n(3 runs)", "direct": "trained on\nrollouts (1 run)"}
colors = {"prompt": "#7f7f7f", "claude": "#2ca02c", "onpolicy": "#1f77b4", "direct": "#d62728"}
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, (n, title) in zip(axes[:2], (("bloom_test", "Bloom test (other models' rollouts)"), ("onpolicy_heldout", "gpt-oss-20b's own rollouts (held out)"))):
    for j, met in enumerate(("auroc", "tpr@1")):
        for i, a in enumerate(arms):
            if n not in S[a]:
                continue
            s = S[a][n]
            ax.bar(j + (i - (len(arms) - 1) / 2) * 0.2, s[f"{met}_mean"], 0.2, yerr=s[f"{met}_sd"] if s["n_runs"] > 1 else None,
                   color=colors[a], label=labels[a] if j == 0 else None, capsize=3)
    ax.set_xticks([0, 1], ["AUROC", "detection at 1%\nfalse alarms"])
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
ax = axes[2]
bb = ["sandbagging", "sycophancy", "sabotage", "deception", "self_preservation"]
for i, a in enumerate(arms):
    pb = S[a]["bloom_test"]["per_behavior_tpr@1_mean"]
    ax.bar(np.arange(len(bb)) + (i - (len(arms) - 1) / 2) * 0.2, [pb.get(b, 0) for b in bb], 0.2, color=colors[a])
ax.set_xticks(range(len(bb)), [b.replace("_", "-") for b in bb], fontsize=8)
ax.set_ylim(0, 1)
ax.set_title("Bloom test, detection at 1% by behavior", fontsize=10)
ax.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(REPO_ROOT / "docs" / "figures" / "seed_source.png", dpi=130)
print("wrote docs/figures/onpolicy_rates.png, seed_source.png")
