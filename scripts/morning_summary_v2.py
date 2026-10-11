"""One readable Markdown summary of a steer_v2 run: latent search, generators, probe tables with intervals,
union / filtered / multi-layer arms, prompt-matched WildChat probes, sycophancy controls.
Every section is optional (missing outputs are reported, not fatal).
Usage: .venv/bin/python scripts/morning_summary_v2.py [pool_dir] [out.md]   (default data/pools/steer_v2)
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from natsteer.config import DATA_ROOT

POOL = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_ROOT / "pools" / "steer_v2"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else POOL / "SUMMARY.md"
lines: list[str] = []


def load(name):
    p = POOL / name
    return json.load(open(p)) if p.exists() else None


def add(*a):
    lines.extend(a)


def section(title, fn):
    add("", f"## {title}", "")
    try:
        fn()
    except Exception as e:  # a missing or malformed output must not stop the summary
        add(f"_could not summarize: {type(e).__name__}: {e}_")


def ci(st, m="tpr@0.01"):
    return f"{st[m]['mean']:.2f} [{st[m]['lo']:.2f},{st[m]['hi']:.2f}]"


def dlt(st, m="tpr@0.01"):
    d = st.get("delta_" + m)
    return f" {d['mean']:+.2f}{'*' if d['lo'] > 0 or d['hi'] < 0 else ''}" if d else ""


def grid_table(f, title, budgets=None, arm_order=None):
    budgets = budgets or list(f)
    arms = sorted(f[budgets[0]]) if budgets and budgets[0] in f else []
    order = arm_order or (["natural"] if "natural" in arms else []) + [a for a in arms if a != "natural"]
    add("", title, "", "| arm | " + " | ".join(f"b={b}" for b in budgets) + " |", "|---|" + "---|" * len(budgets))
    for a in order:
        add(f"| {a} | " + " | ".join((ci(f[b][a]) + dlt(f[b][a])) if b in f and a in f[b] else "-" for b in budgets) + " |")


def timings():
    p = POOL / "timings.log"
    if not p.exists():
        add("_no timings.log_")
        return
    rows = [m.groups() for m in (re.match(r"^(\S+) rc=(\d+) (\d+)s$", l.strip()) for l in open(p)) if m]
    bad = [l.strip() for l in open(p) if "failed" in l or "WATCHDOG" in l]
    add("| stage | rc | seconds |", "|---|---|---|")
    for r in rows:
        add(f"| {r[0]} | {r[1]} | {r[2]} |")
    add("", f"Total {sum(int(r[2]) for r in rows) / 3600:.1f} h. " + (f"Problems: {len(bad)}: " + "; ".join(bad[:4]) if bad else "No failed attempts or watchdog kills."))


def latent_search(s2_name="screen2_table.json", v2_name="latents_validated2_table.json", v1_name="latents_validated_table.json", what="harmless prompts"):
    s2, v2, v1v = load(s2_name), load(v2_name), load(v1_name)
    if not s2:
        add("_no screen2_")
        return
    base = s2["base"]
    ok = lambda s: s["incomplete_rate"] <= base["incomplete_rate"] + 0.15 and s["rep4"] <= base["rep4"] + 0.05  # noqa: E731
    hits = [(c, s) for c, s in s2.items() if s.get("family") == "attr" and ok(s) and s["pos_rate"] >= base["pos_rate"] + 0.2]
    add(f"New screen: {len(s2) - 1} conditions on 64 {what} (baseline {base['pos_rate']:.2f}); {len(hits)} reach base + 0.2 with generation intact.", "")
    add("| layer | strength | conditions | hits |", "|---|---|---|---|")
    keys = sorted({(s["layer"], s["lam"]) for s in s2.values() if s.get("family") == "attr"})
    for L, lam in keys:
        rows = [s for s in s2.values() if s.get("family") == "attr" and s["layer"] == L and s["lam"] == lam]
        add(f"| {L} | {lam} | {len(rows)} | {sum(1 for s in rows if ok(s) and s['pos_rate'] >= base['pos_rate'] + 0.2)} |")
    if v2:
        vb = v2["base"]
        good = sorted([(c, s) for c, s in v2.items() if s.get("family") == "attr" and s["incomplete_rate"] <= vb["incomplete_rate"] + 0.15 and s["pos_rate"] >= vb["pos_rate"] + 0.2], key=lambda cs: -cs[1]["pos_rate"])
        by = {}
        for _, s in good:
            by[s["layer"]] = by.get(s["layer"], 0) + 1
        n1 = sum(1 for s in (v1v or {}).values() if s.get("family") == "attr" and s["pos_rate"] >= (v1v["base"]["pos_rate"] + 0.2))
        add("", f"Validated on fresh prompts: **{len(good)}** distinct latents (v1: {n1}); by layer {by}. Top: " + ", ".join(f"{c} {s['pos_rate']:.2f}" for c, s in good[:12]))
        bund = sorted([(c, s["pos_rate"]) for c, s in v2.items() if s.get("family") == "bundle"], key=lambda x: -x[1])[:4]
        add("", "Best bundles: " + (", ".join(f"{c} {v:.2f}" for c, v in bund) or "none"))


def generators():
    r = load("probes2_report.json")
    if not r:
        add("_no probes2 report_")
        return
    add("| generator | samples | verified positives | final-answer compression | CoT compression |", "|---|---|---|---|---|")
    for g, d in r["generators"].items():
        dv = r["diversity"].get(g, {})
        add(f"| {g} | {d['n']} | {d['verified_pos']} | {dv.get('final_compression', float('nan')):.1f} | {dv.get('cot_compression', float('nan')):.1f} |")
    nd = r["diversity"].get("natural_refusals")
    if nd:
        add(f"| _natural refusals_ | | | {nd['final_compression']:.1f} | {nd['cot_compression']:.1f} |")
    for key, cov in r.get("coverage", {}).items():
        if cov:
            add("", f"Coverage of natural refusals ({key}): precision / recall", "", "| generator | precision | recall |", "|---|---|---|")
            for g, d in cov.items():
                add(f"| {g} | {d['precision']:.2f} | {d['recall']:.2f} |")


def probes2():
    r = load("probes2_report.json")
    if not r:
        add("_no probes2 report_")
        return
    for key in ("L15_cot_first16", "L15_cot_first32", "L15_cot_mean"):
        f = r["features"].get(key)
        if not f:
            continue
        for sl in ("all", "harmless_prompts"):
            if sl in f:
                grid_table(f[sl], f"**{key}, slice {sl}**: TPR at 1% FPR (95% interval; `*` = difference from natural excludes 0)")


def extra():
    r = load("probes2_extra_report.json")
    if not r:
        add("_no probes2_extra report_")
        return
    add("Arms: `union` = positives pooled evenly over the generators; `filtered_*` = the 30% of synthetic positives that a cross-fitted natural-vs-synthetic discriminator finds most natural; `mix_*` = half natural. `multilayer` = layers 11, 15 and 19 concatenated.")
    for variant, d in r["variants"].items():
        for pooling, res in d.items():
            for sl in ("all", "harmless_prompts"):
                if sl in res:
                    grid_table(res[sl], f"**{variant}, {pooling}, slice {sl}**: TPR at 1% FPR")
    for pooling, d in r.get("naturalness", {}).items():
        add("", f"Discriminator score (lower = more natural), {pooling}: " + ", ".join(f"{g} {v['mean_score']:.1f} (kept {v['kept_mean_score']:.1f})" for g, v in d.items()))


def matched():
    r = load("probes_matched_report.json")
    if not r:
        add("_no matched-prompt report_")
        return
    add(f"Held-out WildChat prompts: {r['n_eval']} rows, {r['eval_pos']} natural refusals. Training prompts for the `matched_*` arms are WildChat training prompts the model naturally refuses; `matched_base` positives are the natural refusals on the same prompts, the other `matched_*` positives are steered refusals on them; `*_harmless_prompts` arms are the v1 generators (positives on harmless prompts).", "")
    add("| generator | samples | positives |", "|---|---|---|")
    for g, d in r["matched"].items():
        add(f"| {g} | {d['n']} | {d['positives']} |")
    for key, f in r["features"].items():
        grid_table(f["wildchat_heldout"], f"**{key}**: TPR at 1% FPR (reference: natural WildChat-trained probe)")


def syco2(name="syco_probes2_report.json"):
    r = load(name)
    if not r:
        add("_no sycophancy probes2_")
        return
    ln = r["length"]
    add(f"CoT length (log characters): a length-only probe reaches AUROC {ln['length_only_auroc_heldout']:.2f} on held-out items; natural positives median {ln['natural_pos_median_log_chars']:.1f}, natural negatives {ln['natural_neg_median_log_chars']:.1f}. Generator positives: " + ", ".join(f"{g} {d['pos_median_log_chars']:.1f}" for g, d in ln["generators"].items() if d["pos_median_log_chars"] is not None))
    for key, f in r["features"].items():
        grid_table(f["all"], f"**{key}**: TPR at 1% FPR (`random_dir` = steering with a random direction, labels still from the answer key; `*_lenmatch` = positives resampled to the natural positives' CoT-length deciles)")


def layers():
    r = load("probes_layers_report.json")
    if not r:
        add("_no layer sweep_")
        return
    add("TPR at 1% FPR, natural-trained probe vs synthetic-only (`union` = pooled generators) per layer; `all_layers` concatenates layers 7, 11, 15, 19 and 23. Budget 256, all prompts.", "")
    for pooling, d in r["variants"].items():
        arms = ["natural"] + [f"synthetic_{g}" for g in r["generators"]] + ["synthetic_union", "mix_union"]
        add("", f"**{pooling}**", "", "| arm | " + " | ".join(d) + " |", "|---|" + "---|" * len(d))
        for a in arms:
            cells = []
            for v, res in d.items():
                st = res.get("all", {}).get("256", {}).get(a)
                cells.append(ci(st) if st else "-")
            add(f"| {a} | " + " | ".join(cells) + " |")


add("# steer_v2 morning summary", "", f"Source: `{POOL}`. Intervals are 95% prompt-clustered bootstrap; `*` marks a paired difference from the natural-trained arm whose interval excludes 0.")
section("Stage timings and problems", timings)
section("Sycophancy: does the steering vector matter, and CoT length", syco2)
section("Refusal: wider latent search", latent_search)
section("Sycophancy: wider latent search", lambda: latent_search("syco_screen2_table.json", "syco_latents_validated2_table.json", "syco_latents_validated_table.json", "low-sycophancy items"))
section("Sycophancy: probes with the wider generator pool", lambda: syco2("syco_probes3_report.json"))
section("Layer sweep (7, 11, 15, 19, 23 and all concatenated)", layers)
section("Refusal: generators, diversity, coverage", generators)
section("Refusal: probes with the new generators", probes2)
section("Refusal: union, naturalness-filtered and multi-layer", extra)
section("Prompt-matched positives on held-out WildChat refusals", matched)
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT, f"({len(lines)} lines)")
