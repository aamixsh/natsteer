"""Figures for docs/presentation.md, read from the E0 run reports under data/runs/ and the committed results
under docs/results/ (refusal_v0, steer_v1, steer_v2, steer_e3). CPU only.

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
RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"
REFUSAL = RESULTS / "refusal_v0"
RUNS = {
    "smoke": RUNS_DIR / "e0_smoke/20261008T183341Z/report.json",
    "noise": RUNS_DIR / "e0_noise_floor/20261008T183545Z/report.json",
    "laser": RUNS_DIR / "laser_baseline/aegis_human_rare2pct/20261008T185946Z/report.json",
}
COLORS = {"laser": "#1f77b4", "uncertainty": "#ff7f0e", "random": "#2ca02c"}
# one color per kind of training data, the same in every figure
C = {"natural": "#2ca02c", "prompted": "#1f77b4", "dense": "#ff7f0e", "latent": "#d62728", "bundle": "#9467bd", "control": "#7f7f7f", "mix": "#17becf"}
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
    axes[1].legend(loc="upper right")
    fig.suptitle(f"Aegis human labels, pool at 2% positives ({r['pool_n']:,} prompts), layer 15 user_mean, 5 seeds")
    fig.tight_layout()
    fig.savefig(OUT / "laser_aegis.png", dpi=130)


def _gz(run, name):
    import gzip

    return json.load(gzip.open(RESULTS / run / name))


def _res(run, name):
    return json.load(open(RESULTS / run / name))


def _ci(st, m="tpr@0.01"):
    return st[m]["mean"], st[m]["mean"] - st[m]["lo"], st[m]["hi"] - st[m]["mean"]


def _bars(ax, items, stats, width=0.8, offset=0.0, alpha=1.0, label=None, hatch=None):
    """items: [(arm, color)]; stats: {arm: stat dict}. Bars with 95% interval whiskers."""
    x = np.arange(len(items)) + offset
    v = [_ci(stats[a]) for a, _ in items]
    ax.bar(x, [m for m, _, _ in v], width, yerr=[[lo for _, lo, _ in v], [hi for _, _, hi in v]], capsize=3, color=[c for _, c in items], alpha=alpha, label=label, hatch=hatch, edgecolor="white" if hatch else None)
    return x


def method():
    """Schematic of the pipeline: the steering vector is the policy, the probe reads clean activations."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=11, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=color, ec="#444", lw=1, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1, text=None, dy=0.18):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333"))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", fontsize=9, color="#333")

    # row 1: natural data
    ax.text(0.1, 5.25, "Natural data", fontsize=12, weight="bold", color=C["natural"])
    box(0.1, 4.0, 2.3, 1.0, "prompts\n(harmful, harmless,\nWildChat, opinion items)", "#e8f5e9", fs=9.5)
    box(3.0, 4.0, 2.3, 1.0, "gpt-oss-20b\nsamples (no steering)", "#e8f5e9", fs=9.5)
    box(5.9, 4.0, 2.3, 1.0, "label from the answer\n(refusal phrase or\nanswer key)", "#e8f5e9", fs=9.5)
    arrow(2.4, 4.5, 3.0, 4.5)
    arrow(5.3, 4.5, 5.9, 4.5)
    # row 2: steered data
    ax.text(0.1, 3.05, "Synthetic positives: the steering vector is the policy", fontsize=12, weight="bold", color=C["latent"])
    box(0.1, 1.55, 2.3, 1.25, "behavior / no-behavior\npairs on the same prompt\n-> difference of means\n-> SAE delta-attribution", "#fdecea", fs=9)
    box(3.0, 1.55, 2.3, 1.25, "steer generation\n(dense direction, latents,\nbundles, or a prompt)\non other prompts", "#fdecea", fs=9)
    box(5.9, 1.55, 2.3, 1.25, "keep verified positives\n(same answer-based label)", "#fdecea", fs=9.5)
    arrow(2.4, 2.17, 3.0, 2.17)
    arrow(5.3, 2.17, 5.9, 2.17)
    # shared: clean recompute, probe, evaluation
    box(8.9, 2.5, 2.2, 1.6, "clean forward pass\nof the text\n(no steering)\n-> pooled activations\n(CoT first 8-32 tokens,\nwhole CoT, answer)", "#eef2f7", fs=9)
    arrow(8.2, 4.5, 9.4, 4.1)
    arrow(8.2, 2.17, 9.4, 2.5)
    box(11.7, 3.45, 2.2, 1.1, "logistic probe:\nnatural / synthetic /\nhalf-half, 8-2,048 labels", "#eef2f7", fs=9)
    box(11.7, 1.75, 2.2, 1.25, "evaluate on held-out\nnatural data:\nTPR at 1% FPR,\n95% intervals", "#fff8e1", fs=9.5, bold=True)
    arrow(11.1, 3.5, 11.7, 3.95)
    arrow(12.8, 3.45, 12.8, 3.0)
    ax.text(7.0, 0.55, "Steering only produces text; the probe never sees steered activations.  Labels never use the steering vector.\n"
            "Controls: random directions, random and activation-difference latents, projection of the direction out of the probe.", ha="center", fontsize=10, color="#333")
    fig.tight_layout()
    fig.savefig(OUT / "method.png", dpi=130)


def refusal_steering():
    """(a) dense-direction strength sweep, (b) latents that work by strength, (c) where steering acts."""
    sw = json.load(open(REFUSAL / "sweep_table.json"))
    v2 = _res("steer_v2", "latents_validated2_table.json")
    v1s = _gz("steer_v1", "latents_screen_table.json.gz")
    tm = _res("steer_v1", "timing_table.json")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    for kind, L, col, lab in (("eop", 15, C["dense"], "end-of-prompt direction, L15"), ("cot", 11, "#8c564b", "CoT direction, L11"), ("random", 15, C["control"], "random direction, L15")):
        pts = sorted((s["coeff"], s["refusal_rate"], s["incomplete_rate"]) for s in sw.values() if s.get("kind") == kind and s.get("layer") == L and s.get("positions") == "all")
        a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=lab)
        if kind != "random":
            a.plot([p[0] for p in pts], [p[2] for p in pts], "x:", color=col, alpha=0.7)
    a.set_xlabel("strength (multiples of the difference of means)")
    a.set_ylabel("share of samples (harmless prompts)")
    a.set_title("(a) Dense direction: narrow window", fontsize=11)
    a.text(0.02, 0.55, "solid: refusal\ndotted: cut off (no answer)", transform=a.transAxes, fontsize=9)
    a.legend(fontsize=8.5, loc="center right")
    a.grid(alpha=0.3)
    v1 = _res("steer_v1", "latents_validated_table.json")

    def working(t):
        bs = t["base"]
        return [s for s in t.values() if s.get("family") == "attr" and s["pos_rate"] >= bs["pos_rate"] + 0.2 and s["incomplete_rate"] <= bs["incomplete_rate"] + 0.15]

    passes = [("first pass\ntop 32 per layer\nstrength 2 and 5", working(v1)), ("wider pass\ntop 128 per layer\nstrength 5, 8 and 12", working(v2))]
    bottom = np.zeros(2)
    for L, col in ((11, "#d62728"), (15, "#ff9896"), (19, "#7f7f7f")):
        n = np.array([sum(1 for s in g if s["layer"] == L) for _, g in passes])
        b.bar(range(2), n, 0.6, bottom=bottom, color=col, label=f"layer {L}")
        bottom = bottom + n
    for i, t in enumerate(bottom):
        b.text(i, t + 0.4, f"{int(t)}", ha="center", fontsize=12, weight="bold")
    ctl = [s for s in v1s.values() if s.get("family") in ("act", "randlat")]
    b.set_xticks(range(2), [n for n, _ in passes], fontsize=9.5)
    b.set_ylabel("single latents that refuse >= 20%\non 96 fresh prompts, text intact")
    b.set_title("(b) Delta-attributed SAE latents", fontsize=11)
    b.text(0.03, 0.95, f"controls: activation-difference and\nrandom latents work 0 of {len(ctl)}", transform=b.transAxes, fontsize=9, va="top")
    b.legend(fontsize=9, loc="center left")
    b.set_ylim(0, bottom.max() + 4)
    b.grid(alpha=0.3, axis="y")
    by = {}
    for s in tm.values():
        by.setdefault(s["kind"], {})[s["positions"]] = s["pos_rate"]
    rows = [("eop_L15_all_c2.0", "dense direction"), ("valbundle_L11_k3_x2.0", "3-latent bundle"), ("attr_L11_34425_x5.0", "latent 34425"), ("attr_L15_68915_x5.0", "latent 68915")]
    x = np.arange(len(rows))
    for j, (pos, col, lab) in enumerate((("prompt", "#c7c7c7", "prompt tokens only"), ("cot", "#d62728", "CoT tokens only"), ("final", "#ffbb78", "answer tokens only"))):
        c.bar(x + (j - 1) * 0.27, [by[k].get(pos, np.nan) for k, _ in rows], 0.27, color=col, label=lab)
    c.set_xticks(x, [n for _, n in rows], fontsize=9.5)
    c.set_ylabel("refusal rate (32 prompts)")
    c.set_title("(c) Steering acts through the chain of thought", fontsize=11)
    c.legend(fontsize=9)
    c.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "refusal_steering.png", dpi=130)


def refusal_probes():
    """Every generator against natural labels: early-CoT features separate them, whole-CoT features saturate."""
    r = _gz("steer_v2", "probes2_report.json.gz")["features"]
    items = [("natural", C["natural"], "natural"), ("synthetic_dom_all", C["dense"], "dense\ndirection"), ("synthetic_dense_multi", C["dense"], "dense,\n3 settings"), ("synthetic_prompted", C["prompted"], "prompted"), ("synthetic_valbundle2", C["bundle"], "4-latent\nbundle"), ("synthetic_many_bundles", C["bundle"], "random 3-latent\nbundles"), ("synthetic_many_mix", C["latent"], "23 latents,\none each"), ("mix_dense_multi", C["mix"], "half natural,\nhalf dense")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for j, (bud, alpha) in enumerate((("64", 0.55), ("256", 1.0))):
        x = _bars(a, [(k, c) for k, c, _ in items], r["L15_cot_first32"]["all"][bud], width=0.38, offset=(j - 0.5) * 0.4, alpha=alpha)
    a.set_xticks(np.arange(len(items)), [n for _, _, n in items], fontsize=9)
    a.axhline(r["L15_cot_first32"]["all"]["256"]["natural"]["tpr@0.01"]["mean"], color=C["natural"], ls="--", lw=1)
    a.set_ylabel("TPR at 1% FPR (held-out natural)")
    a.set_title("First 32 CoT tokens, layer 15: light = 64 labels, dark = 256 labels", fontsize=11)
    a.set_ylim(0, 0.85)
    a.grid(alpha=0.3, axis="y")
    _bars(b, [(k, c) for k, c, _ in items], r["L15_cot_mean"]["all"]["64"], width=0.7)
    b.set_xticks(np.arange(len(items)), [n.replace("\n", " ") for _, _, n in items], fontsize=8.5, rotation=40, ha="right")
    b.set_ylim(0.8, 1.0)
    b.set_title("Whole CoT, 64 labels: every arm is saturated", fontsize=11)
    b.grid(alpha=0.3, axis="y")
    fig.suptitle("Refusal probes: 10 seeds, 95% prompt-clustered intervals, 3,000 prompts per generator", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "refusal_probes.png", dpi=130)


def refusal_topic():
    """(a) same prompts as natural refusals close the gap, (b) keeping natural-looking synthetic text helps."""
    m = _gz("steer_v2", "probes_matched_report.json.gz")["features"]["L15_cot_first32"]["wildchat_heldout"]["128"]
    ex = _gz("steer_v2", "probes2_extra_report.json.gz")["variants"]["single"]["cot_first32"]["all"]["256"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    items = [("natural", C["natural"], "natural"), ("synthetic_matched_prompted", C["prompted"], "prompted"), ("synthetic_matched_latent", C["latent"], "2 latents"), ("synthetic_matched_dom_all", C["dense"], "dense"), ("synthetic_matched_valbundle", C["bundle"], "bundle"), ("synthetic_prompted_harmless_prompts", C["prompted"], "prompted"), ("synthetic_dom_all_harmless_prompts", C["dense"], "dense"), ("synthetic_latent_mix_harmless_prompts", C["latent"], "4 latents")]
    x = _bars(a, [(k, c) for k, c, _ in items], m, width=0.72)
    for i in range(5, 8):
        a.patches[i].set_hatch("//")
        a.patches[i].set_edgecolor("white")
    a.set_xticks(x, [n for _, _, n in items], fontsize=9.5)
    a.axhline(m["natural"]["tpr@0.01"]["mean"], color=C["natural"], ls="--", lw=1)
    a.axvspan(0.55, 4.45, color="#e3f2fd", alpha=0.6, zorder=0)
    a.text(2.5, 0.66, "steered on the same prompts\nthe model naturally refuses", ha="center", fontsize=10)
    a.text(6, 0.66, "steered on\nharmless prompts", ha="center", fontsize=10)
    a.set_ylim(0, 0.75)
    a.set_ylabel("TPR at 1% FPR")
    a.set_title("(a) Held-out WildChat refusals, first 32 CoT tokens, 128 labels", fontsize=11)
    a.grid(alpha=0.3, axis="y")
    pairs = [("synthetic_union", "synthetic_filtered_union", "all generators pooled", C["bundle"]), ("synthetic_dom_all", "synthetic_filtered_dom_all", "dense direction", C["dense"]), ("synthetic_prompted", "synthetic_filtered_prompted", "prompted", C["prompted"])]
    x = np.arange(len(pairs))
    for j, (idx, alpha, lab) in enumerate(((0, 0.45, "all synthetic positives"), (1, 1.0, "30% most natural-looking"))):
        v = [_ci(ex[p[idx]]) for p in pairs if p[idx] in ex]
        b.bar(x + (j - 0.5) * 0.38, [t[0] for t in v], 0.38, yerr=[[t[1] for t in v], [t[2] for t in v]], capsize=3, color=[p[3] for p in pairs], alpha=alpha, label=lab)
    b.axhline(ex["natural"]["tpr@0.01"]["mean"], color=C["natural"], ls="--", lw=1)
    b.text(2.45, ex["natural"]["tpr@0.01"]["mean"] + 0.015, "natural", color=C["natural"], ha="right", fontsize=10)
    b.set_xticks(x, [p[2] for p in pairs], fontsize=10)
    b.set_ylim(0, 0.8)
    b.set_ylabel("TPR at 1% FPR")
    b.set_title("(b) Refusal test set, 256 labels: filter by a\nnatural-vs-synthetic discriminator (cross-fitted)", fontsize=11)
    b.legend(fontsize=9, loc="upper left")
    b.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "refusal_topic.png", dpi=130)


def refusal_layers():
    """(a) the gap is largest at the layer where natural probes are best, (b) latent steering stays near the
    natural activations, yet that does not make its data more useful."""
    lay = _gz("steer_v2", "probes_layers_report.json.gz")["variants"]["cot_first32"]
    reach = _res("steer_v1", "reach_table.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    layers = np.array([7, 11, 15, 19, 23])
    for k, (arm, label, c, st) in enumerate((("natural", "natural", C["natural"], "o-"), ("synthetic_dom_all", "dense direction", C["dense"], "^-"), ("synthetic_prompted", "prompted", C["prompted"], "v-"), ("synthetic_union", "all generators pooled", C["bundle"], "s-"), ("synthetic_many_mix", "23 latents", C["latent"], "x-"))):
        st_ = [lay[f"L{L}"]["all"]["256"][arm]["tpr@0.01"] for L in layers]
        a.errorbar(layers + (k - 2) * 0.25, [s["mean"] for s in st_], yerr=[[s["mean"] - s["lo"] for s in st_], [s["hi"] - s["mean"] for s in st_]], fmt=st, color=c, capsize=2, label=label)
    a.set_xticks(layers)
    a.set_xlabel("layer the probe reads")
    a.set_ylabel("TPR at 1% FPR")
    a.set_title("(a) Probe layer: refusal test set, first 32 CoT tokens, 256 labels", fontsize=11)
    a.legend(fontsize=9, loc="lower left", ncol=2)
    a.grid(alpha=0.3)
    groups = {"single latent": [], "bundle": [], "dense direction": []}
    for e in reach:
        g = "dense direction" if e["family"] in ("dom_all", "dom_prompt", "random_dir") else ("bundle" if e["family"] in ("bundle", "valbundle", "random_bundles") else "single latent")
        groups[g].append(e)
    floor = np.median([e["natural_floor"]["rel_mse"] for e in reach])
    names = ["natural text", "single latent", "bundle", "dense direction"]
    vals = [floor] + [np.median([e["rel_mse_steered"] for e in groups[g]]) for g in names[1:]]
    b.bar(range(4), vals, color=[C["natural"], C["latent"], C["bundle"], C["dense"]])
    for i, v in enumerate(vals):
        b.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=10)
    b.set_xticks(range(4), names, fontsize=10)
    b.set_ylabel("SAE reconstruction error (relative MSE)")
    b.set_title("(b) Steered activations during generation:\nlatents stay close to natural, the dense direction does not", fontsize=11)
    b.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "refusal_layers.png", dpi=130)


def laser_warmstart():
    """Free synthetic positives as a LASER warm start on benign WildChat traffic (4.7% natural refusals)."""
    r = _res("steer_v1", "laser_warmstart.json")["keys"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    arms = (("laser", "LASER, natural labels only", C["natural"], "o-"), ("laser+dom_all", "LASER + 64 dense-direction positives", C["dense"], "^-"), ("laser+prompted", "LASER + 64 prompted positives", C["prompted"], "v-"), ("laser+latent_mix", "LASER + 64 latent positives", C["latent"], "x-"), ("random", "random sampling", C["control"], "s--"))
    for ax, key, title in zip(axes, ("L15_cot_mean", "L15_cot_first32"), ("whole CoT", "first 32 CoT tokens")):
        for arm, lab, c, st in arms:
            rows = r[key][arm]
            ax.plot([p["n_labels"] for p in rows], [p["tpr@0.01"]["mean"] for p in rows], st, color=c, label=lab, ms=4)
        ax.set_xlabel("natural labels bought")
        ax.set_title(f"{title}, layer 15", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TPR at 1% FPR (held-out WildChat)")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Active learning on benign WildChat traffic (4.7% natural refusals), 5 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "laser_warmstart.png", dpi=130)


def syco():
    """(a) steering raises sycophancy, but so do random latents, (b) probes: synthetic plateaus, a random
    direction is as good as the chosen ones."""
    sw = _res("steer_v1", "syco_sweep_table.json")
    s2 = _gz("steer_v2", "syco_screen2_table.json.gz")
    s1 = _gz("steer_v1", "syco_latents_screen_table.json.gz")
    f = _gz("steer_v2", "syco_probes3_report.json.gz")["features"]["L15_final_mean"]["all"]
    base = sw["base"]["pos_rate"]
    best_dense = max((s["pos_rate"], c) for c, s in sw.items() if s.get("family") == "dom" and s["incomplete_rate"] <= sw["base"]["incomplete_rate"] + 0.15)
    rnd_dir = max(s["pos_rate"] for s in sw.values() if s.get("family") == "random")
    ok = lambda s: s["incomplete_rate"] <= s2["base"]["incomplete_rate"] + 0.15  # noqa: E731
    attr = [s for t in (s1, s2) for s in t.values() if s.get("family") == "attr"]
    best_lat = max(s["pos_rate"] for s in s2.values() if s.get("family") == "attr" and ok(s))
    n_dom = sum(1 for s in sw.values() if s.get("family") == "dom")
    n_rnd = sum(1 for s in sw.values() if s.get("family") == "random")
    rnd_lat = max(s["pos_rate"] for s in s1.values() if s.get("family") == "randlat" and s["incomplete_rate"] <= s1["base"]["incomplete_rate"] + 0.15)
    n_rl = sum(1 for s in s1.values() if s.get("family") == "randlat")
    bars = [("unsteered", base, "#c7c7c7"), (f"random\ndirection\n(best of {n_rnd})", rnd_dir, C["control"]), (f"random\nlatent\n(best of {n_rl})", rnd_lat, C["control"]), (f"dense\ndirection\n(best of {n_dom})", best_dense[0], C["dense"]), (f"attributed\nlatent\n(best of {len(attr)})", best_lat, C["latent"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    a.bar(range(len(bars)), [v for _, v, _ in bars], color=[c for _, _, c in bars])
    for i, (_, v, _) in enumerate(bars):
        a.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=10)
    a.set_xticks(range(len(bars)), [n for n, _, _ in bars], fontsize=9.5)
    a.set_ylabel("sycophantic answers (64 low-sycophancy items)")
    a.set_ylim(0, 1)
    a.set_title("(a) Steering toward sycophancy: best setting of each kind\n(screening items; best-of-N values are optimistic)", fontsize=11)
    a.grid(alpha=0.3, axis="y")
    budgets = sorted(map(int, f))
    for arm, label, c, st in (("natural", "natural", C["natural"], "o-"), ("synthetic_many_mix", "39 latents, one per item group", C["latent"], "s-"), ("synthetic_dom_all", "dense CoT direction", C["dense"], "^-"), ("synthetic_random_dir", "random direction (control)", C["control"], "x--"), ("mix_union", "half natural, half synthetic", C["mix"], "d-")):
        bs = [x for x in budgets if arm in f[str(x)]]
        mm = np.array([f[str(x)][arm]["tpr@0.01"]["mean"] for x in bs])
        lo = np.array([f[str(x)][arm]["tpr@0.01"]["lo"] for x in bs])
        hi = np.array([f[str(x)][arm]["tpr@0.01"]["hi"] for x in bs])
        b.plot(bs, mm, st, color=c, label=label)
        b.fill_between(bs, lo, hi, color=c, alpha=0.08)
    b.set_xscale("log")
    b.minorticks_off()
    b.set_xticks(budgets, [str(x) for x in budgets])
    b.set_xlabel("training labels")
    b.set_ylabel("TPR at 1% FPR (held-out natural)")
    b.set_title("(b) Probes on the final answer, layer 15 (10 seeds, 95% intervals)", fontsize=11)
    b.legend(fontsize=9, loc="upper left")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "syco.png", dpi=130)


def e2_reach():
    """E2 at scale: (a) how far steering moves activations from natural, against the behavior it produces;
    (b) per refusal latent: yield against the usefulness of its data for probes, colored by distance."""
    t = _res("steer_e3", "reach2_table.json")
    u = _res("steer_e3", "e2_usefulness.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.6))
    for task, col, lab in (("refusal", C["latent"], "refusal latents"), ("syco", "#8c564b", "sycophancy latents")):
        es = [e for e in t if e["task"] == task and e["family"] == "attr" and e["pos_rate"] is not None]
        a.scatter([max(e["rel_mse_steered"] - e["rel_mse_unsteered"], 1e-4) for e in es], [e["pos_rate"] - e["base_rate"] for e in es], color=col, label=f"{lab} (n = {len(es)})", alpha=0.8)
    a.set_xscale("log")
    a.set_xlabel("increase in SAE reconstruction error when steered (log)")
    a.set_ylabel("behavior gain over unsteered (fresh prompts)")
    cr, cs = u["correlations"]["refusal:rel_mse_steered_increase~yield"], u["correlations"]["syco:rel_mse_steered_increase~yield"]
    a.set_title(f"(a) Distance from natural vs effect: refusal Spearman {cr['spearman']:.2f}, sycophancy {cs['spearman']:.2f}", fontsize=10.5)
    a.legend(fontsize=9)
    a.grid(alpha=0.3)
    rows = u["latents"]
    sc = b.scatter([r["yield"] for r in rows], [r["tpr"] for r in rows], c=[np.log10(max(r["recon_increase"], 1e-3)) for r in rows], cmap="viridis", s=60)
    cb = fig.colorbar(sc, ax=b)
    cb.set_label("log10 reconstruction-error increase")
    b.axhline(u["natural"]["tpr@0.01"]["mean"], color=C["natural"], ls="--", lw=1)
    b.text(0.45, u["natural"]["tpr@0.01"]["mean"] + 0.01, "natural labels", color=C["natural"], fontsize=10)
    c1, c2 = u["correlations"]["yield~tpr"], u["correlations"]["recon_increase~tpr"]
    b.set_xlabel("refusal yield of the latent (fresh prompts)")
    b.set_ylabel(f"TPR at 1% FPR, probe on its data ({u['budget']} labels)")
    b.set_title(f"(b) Usefulness of each latent's data: yield Spearman {c1['spearman']:.2f}, distance {c2['spearman']:.2f}", fontsize=10.5)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "e2_reach.png", dpi=130)


def e3():
    """E3: best-of-8 selection by a discriminator and per-token capped steering, against random-of-8 and natural."""
    import gzip

    r = json.load(gzip.open(RESULTS / "steer_e3" / "e3_probes_report.json.gz"))
    ts = r["test_set"]["L15_cot_first32"]["all"]["256"]
    wc = r["wildchat"]["L15_cot_first32"]["wildchat_heldout"]["128"]
    fams = [("dense", C["dense"], "k_dense", ["cap_dense_c3.0"]), ("latents", C["latent"], "k_latents", ["cap_latents_x1.0", "cap_latents_x1.5"]), ("prompted", C["prompted"], "k_prompted", [])]
    variants = [("_rand", "random of 8", 0.35, None), ("_best_label", "best of 8, labeled discriminator", 1.0, None), ("_best_free", "best of 8, label-free", 0.6, "..")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6))
    for ax, res, title in ((axes[0], ts, "(a) Harmless prompts, refusal test set, 256 labels"), (axes[1], wc, "(b) Naturally refused WildChat prompts, held-out WildChat, 128 labels")):
        x0 = 0
        ticks, labels = [], []
        for fam, col, k, caps in fams:
            arms = [("synthetic_" + k + v, lab, al, h) for v, lab, al, h in variants] + [("synthetic_" + c, "capped " + c.split("_")[-1], 1.0, "//") for c in caps]
            for j, (arm, lab, al, h) in enumerate(arms):
                if arm not in res:
                    continue
                m, lo, hi = _ci(res[arm])
                ax.bar(x0 + j, m, 0.85, yerr=[[lo], [hi]], capsize=2, color=col, alpha=al, hatch=h, edgecolor="white" if h else None)
            ticks.append(x0 + (len(arms) - 1) / 2)
            labels.append(fam)
            x0 += len(arms) + 1
        ax.axhline(res["natural"]["tpr@0.01"]["mean"], color=C["natural"], ls="--", lw=1.2)
        ax.text(x0 - 1.2, res["natural"]["tpr@0.01"]["mean"] + 0.012, "natural", color=C["natural"], ha="right", fontsize=10)
        ax.set_xticks(ticks, labels, fontsize=11)
        ax.set_ylabel("TPR at 1% FPR (first 32 CoT tokens)")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 0.8)
        ax.grid(alpha=0.3, axis="y")
    from matplotlib.patches import Patch

    axes[1].legend(handles=[Patch(color="#555", alpha=0.35, label="random of 8"), Patch(color="#555", alpha=1.0, label="best of 8, labeled discriminator"), Patch(facecolor="#555", alpha=0.6, hatch="..", edgecolor="white", label="best of 8, label-free"), Patch(facecolor="#555", hatch="//", edgecolor="white", label="capped (E3a): x1, x1.5 / dense c3")], fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "e3.png", dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (method, noise_floor, laser_curves, refusal_steering, refusal_probes, refusal_topic, refusal_layers, laser_warmstart, syco, e2_reach, e3):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
