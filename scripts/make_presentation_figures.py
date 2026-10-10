"""Figures for docs/presentation.md, read from the run reports under data/runs/ and the committed
refusal results under docs/results/refusal_v0 (CPU only).

Usage: .venv/bin/python scripts/make_presentation_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from natsteer.config import RUNS_DIR

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
REFUSAL = Path(__file__).resolve().parents[1] / "docs" / "results" / "refusal_v0"
RUNS = {
    "smoke": RUNS_DIR / "e0_smoke/20261008T183341Z/report.json",
    "noise": RUNS_DIR / "e0_noise_floor/20261008T183545Z/report.json",
    "laser": RUNS_DIR / "laser_baseline/aegis_human_rare2pct/20261008T185946Z/report.json",
    "syco": RUNS_DIR / "syco_pipeline/20261008T215319Z/report.json",
}
COLORS = {"laser": "#1f77b4", "uncertainty": "#ff7f0e", "random": "#2ca02c"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    return json.load(open(RUNS[name]))


def noise_floor():
    r = load("noise")
    layers = sorted(map(int, r))
    med = [100 * r[str(L)]["rel_diff_median"] for L in layers]
    p99 = [100 * r[str(L)]["rel_diff_p99"] for L in layers]
    flip = [100 * r[str(L)]["routing_flip_rate"] for L in layers]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(layers, med, "o-", label="median")
    a.plot(layers, p99, "s-", label="99th percentile")
    a.set_xlabel("layer")
    a.set_ylabel("relative difference (%)")
    a.set_title("Same prefix: alone vs inside a longer sequence")
    a.legend()
    a.grid(alpha=0.3)
    b.bar(layers, flip, color="#888")
    b.set_xlabel("layer")
    b.set_ylabel("tokens with a different top-4 expert set (%)")
    b.set_title("MoE routing flips")
    b.grid(alpha=0.3, axis="y")
    for ax in (a, b):
        for L in (15, 19):
            ax.axvline(L, color="red", alpha=0.25, lw=6)
    fig.tight_layout()
    fig.savefig(OUT / "noise_floor.png", dpi=130)


def sae_fve():
    r = load("smoke")["sae"]
    names = [("andyrdt_L15_k64", "L15 k=64"), ("andyrdt_L15_k128", "L15 k=128"), ("andyrdt_L19_k64", "L19 k=64"), ("andyrdt_L19_k128", "L19 k=128"), ("goodfire_L15_k32_subtract_bdec=True", "Goodfire L15")]
    segs = ["user", "cot", "final"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, s in enumerate(segs):
        ax.bar(x + (i - 1) * 0.26, [r[n][s]["frac_variance_explained"] for n, _ in names], 0.26, label={"user": "user prompt", "cot": "chain of thought", "final": "final answer"}[s])
    ax.axhline(0.764, color="k", ls="--", lw=1)
    ax.text(len(names) - 0.5, 0.775, "published L15 k=64: 0.76", ha="right", fontsize=10)
    ax.set_xticks(x, [l for _, l in names])
    ax.set_ylabel("fraction of variance explained")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("SAE reconstruction by segment (100 XSTest conversations, 93k tokens)")
    fig.tight_layout()
    fig.savefig(OUT / "sae_fve.png", dpi=130)


def laser_curves():
    r = load("laser")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in zip(axes, ("tpr@0.01", "n_pos"), ("TPR at 1% FPR (Aegis test)", "positives found in the pool")):
        for arm, runs in r["curves"].items():
            x = [p["n_labels"] for p in runs[0]]
            v = np.array([[p.get(metric, np.nan) for p in run] for run in runs], dtype=float)
            m, sd = np.nanmean(v, 0), np.nanstd(v, 0)
            ax.plot(x, m, label={"laser": "LASER", "uncertainty": "uncertainty only", "random": "random"}[arm], color=COLORS[arm])
            ax.fill_between(x, m - sd, m + sd, alpha=0.15, color=COLORS[arm])
        ax.set_xlabel("labels")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ceiling = r["all_label_sweep"]["L15_user_mean"]["tpr@0.01"]
    axes[0].axhline(ceiling, color="k", ls="--", lw=1)
    axes[0].text(2000, ceiling + 0.01, f"all {r['pool_n']:,} labels", ha="right", fontsize=10)
    axes[1].axhline(r["pool_pos"], color="k", ls="--", lw=1)
    axes[1].text(100, r["pool_pos"] - 15, f"all {r['pool_pos']} positives", fontsize=10)
    axes[0].legend()
    fig.suptitle(f"Aegis human labels, pool at 2% positives ({r['pool_n']:,} prompts), layer 15 user_mean, 5 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "laser_aegis.png", dpi=130)


def syco_steering():
    r = load("syco")
    conds = ["prompt", "final", "cot", "all"]
    labels = ["prompt only", "final only", "CoT only", "all positions"]
    n = 96
    natural = [json.loads(l)["label"] for l in open(RUNS_DIR.parent / "pools/syco_v0/train.jsonl")]
    natural_unparsed = 100 * natural.count(None) / len(natural)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(conds))
    axes[0].bar(x - 0.2, [100 * r["steering"][c]["same_items_natural_rate"] for c in conds], 0.4, label="same items, unsteered", color="#bbb")
    axes[0].bar(x + 0.2, [100 * r["steering"][c]["sycophancy_rate"] for c in conds], 0.4, label="steered", color="#d62728")
    axes[0].set_ylim(50, 85)
    axes[0].set_ylabel("sycophantic answers (%)")
    axes[0].set_title("Effect")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[1].bar(x, [100 * r["steering_unparsed"][c] / n for c in conds], color="#9467bd")
    axes[1].axhline(natural_unparsed, color="k", ls="--", lw=1)
    axes[1].text(-0.4, natural_unparsed + 1.5, f"natural: {natural_unparsed:.0f}%", fontsize=10)
    axes[1].set_ylabel("answers with no clear choice (%)")
    axes[1].set_title("Damage")
    axes[2].bar(x, [r["discriminator_auc"][c] for c in conds], color="#17becf")
    axes[2].axhline(0.5, color="k", ls="--", lw=1)
    axes[2].text(-0.4, 0.52, "indistinguishable", fontsize=10)
    axes[2].set_ylim(0.4, 1.02)
    axes[2].set_ylabel("AUC: steered vs natural")
    axes[2].set_title("Unnaturalness (clean activations)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Sycophancy proxy: diff-of-means direction at layer 15, λ = 4, 96 items per condition")
    fig.tight_layout()
    fig.savefig(OUT / "syco_steering.png", dpi=130)


def syco_budget():
    r = load("syco")["arms"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, label, c in (("natural_random", "natural, random", "#2ca02c"), ("natural_laser", "natural, LASER", "#1f77b4"), ("natural_plus_steered", "half natural + half steered", "#d62728")):
        b = sorted(map(int, r[arm]))
        ax.plot(b, [r[arm][str(x)]["auroc"] for x in b], "o-", label=label, color=c)
    ax.set_xlabel("labels")
    ax.set_ylabel("AUROC on held-out natural answers")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([50, 100, 200, 400], ["50", "100", "200", "400"])
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Probe at equal label budgets (L15 final_mean, 5 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "syco_budget.png", dpi=130)


def _refusal(name):
    return json.load(open(REFUSAL / name))


def refusal_steering():
    t = _refusal("sweep_table.json")
    base = t["base"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, pos, title in ((axes[0], "all", "all positions"), (axes[1], "prompt", "prompt tokens only")):
        for kind, L, col in (("eop", 15, "#1f77b4"), ("cot", 11, "#d62728"), ("eop", 11, "#7f7f7f")):
            pts = sorted((s["coeff"], s["refusal_rate"], s["incomplete_rate"]) for s in t.values() if s.get("kind") == kind and s.get("layer") == L and s.get("positions") == pos)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=f"{kind} direction, L{L}: refusal")
            ax.plot([p[0] for p in pts], [p[2] for p in pts], "x--", color=col, alpha=0.6)
        rnd = sorted((s["coeff"], s["refusal_rate"]) for s in t.values() if s.get("kind") == "random" and s.get("layer") == 15 and s.get("positions") == pos)
        ax.plot([p[0] for p in rnd], [p[1] for p in rnd], "s:", color="k", label="random direction, L15: refusal")
        ax.set_title(f"Steering {title}")
        ax.set_xlabel("coefficient (multiples of the difference of means)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of 192 samples on harmless prompts")
    axes[0].legend(fontsize=9, loc="center left")
    axes[1].text(0.02, 0.95, "dashed x: incomplete (no final answer)", transform=axes[1].transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "refusal_steering.png", dpi=130)


def refusal_probes():
    r = _refusal("probes_report.json")["features"]
    arms = [("natural", "natural"), ("synthetic_dom_all", "difference of\nmeans"), ("synthetic_prompted", "prompted"), ("synthetic_bundle", "latent\nbundle"), ("synthetic_diverse", "2-latent\nmix")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (key, sl, title) in zip(axes, (("L15_cot_mean", "harmful_prompts", "whole CoT, natural refusals on harmful prompts"), ("L15_cot_first32", "harmful_prompts", "first 32 CoT tokens, harmful prompts"), ("L15_cot_first32", "harmless_prompts", "first 32 CoT tokens, harmless prompts"))):
        x = np.arange(len(arms))
        for j, (b, col) in enumerate((("64", "#9ecae1"), ("256", "#3182bd"))):
            vals = [r[key]["arms"][a][b][sl]["tpr@0.01"] for a, _ in arms]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=col, label=f"{b} labels")
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in arms], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("TPR at 1% FPR (natural held-out)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "refusal_probes.png", dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (noise_floor, sae_fve, laser_curves, syco_steering, syco_budget, refusal_steering, refusal_probes):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
