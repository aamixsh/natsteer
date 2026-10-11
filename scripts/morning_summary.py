"""One readable Markdown summary of a steer_v1 run (and its extras): stage timings and failures, latent
search, generator yields and diversity, probe tables with intervals, sycophancy, LASER warm start, WildChat
probes, reachability and steering timing. Every section is optional: missing outputs are reported, not fatal.
Usage: .venv/bin/python scripts/morning_summary.py [pool_dir] [out.md]
Default pool_dir is data/pools/steer_v1; output goes next to it as SUMMARY.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

from natsteer.config import DATA_ROOT

POOL = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_ROOT / "pools" / "steer_v1"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else POOL / "SUMMARY.md"
lines: list[str] = []


def load(name):
    p = POOL / name
    return json.load(open(p)) if p.exists() else None


def jl(name):
    p = POOL / name
    return [json.loads(x) for x in open(p)] if p.exists() else []


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
    if not d:
        return ""
    return f" {d['mean']:+.2f}{'*' if d['lo'] > 0 or d['hi'] < 0 else ''}"


def timings():
    p = POOL / "timings.log"
    if not p.exists():
        add("_no timings.log_")
        return
    import re

    rows = [m.groups() for m in (re.match(r"^(\S+) rc=(\d+) (\d+)s$", l.strip()) for l in open(p)) if m]
    bad = [l.strip() for l in open(p) if "failed" in l or "WATCHDOG" in l]
    add("| stage | rc | seconds |", "|---|---|---|")
    for r in rows:
        add(f"| {r[0]} | {r[1]} | {r[2]} |")
    total = sum(int(r[2]) for r in rows)
    add("", f"Total {total / 3600:.1f} h. " + (f"Problems: {len(bad)} (see timings.log): " + "; ".join(bad[:4]) if bad else "No failed attempts or watchdog kills."))


def latent_search(prefix, label):
    scr = load(f"{prefix}latents_screen_table.json")
    val = load(f"{prefix}latents_validated_table.json")
    if not scr:
        add("_no screen_")
        return
    base = scr["base"]
    ok = lambda s: s["incomplete_rate"] <= base["incomplete_rate"] + 0.15 and s["rep4"] <= base["rep4"] + 0.05  # noqa: E731
    add(f"Screen: {len(scr) - 1} conditions on 64 {label}; baseline positive rate {base['pos_rate']:.2f}.", "", "| family | conditions | positive rate >= base+0.2 (generation intact) | best |", "|---|---|---|---|")
    for fam in ("attr", "act", "randlat"):
        rows = [(c, s) for c, s in scr.items() if s.get("family") == fam]
        good = [(c, s) for c, s in rows if ok(s) and s["pos_rate"] >= base["pos_rate"] + 0.2]
        best = max(((c, s["pos_rate"]) for c, s in rows if ok(s)), key=lambda x: x[1], default=("-", 0))
        add(f"| {fam} | {len(rows)} | {len(good)} | {best[0]} {best[1]:.2f} |")
    by_layer = {}
    for c, s in scr.items():
        if s.get("family") == "attr" and ok(s) and s["pos_rate"] >= base["pos_rate"] + 0.2:
            by_layer.setdefault(s["layer"], set()).add(s["latents"][0])
    add("", "Distinct attributed latents that pass the screen, by layer: " + (", ".join(f"L{L}: {len(v)}" for L, v in sorted(by_layer.items())) or "none"))
    if val:
        vb = val["base"]
        good = sorted([(c, s) for c, s in val.items() if s.get("family") == "attr" and s["incomplete_rate"] <= vb["incomplete_rate"] + 0.15 and s["pos_rate"] >= vb["pos_rate"] + 0.2], key=lambda cs: -cs[1]["pos_rate"])
        n_att = sum(1 for s in val.values() if s.get("family") == "attr")
        add("", f"Validation on fresh prompts: {len(good)} of {n_att} latents still pass. Top: " + ", ".join(f"{c} {s['pos_rate']:.2f}" for c, s in good[:8]))
        bund = sorted([(c, s["pos_rate"]) for c, s in val.items() if s.get("family") == "bundle"], key=lambda x: -x[1])[:3]
        add("", "Best bundles: " + (", ".join(f"{c} {v:.2f}" for c, v in bund) or "none"))


def refusal_generators_and_probes(prefix, report_name, key_list, slices, pos_name):
    r = load(report_name)
    if not r:
        add("_no probes report_")
        return
    add("| generator | samples | verified positives | final-answer compression | CoT compression |", "|---|---|---|---|---|")
    for g, d in r["generators"].items():
        dv = r["diversity"].get(g, {})
        add(f"| {g} | {d['n']} | {d['verified_pos']} | {dv.get('final_compression', float('nan')):.1f} | {dv.get('cot_compression', float('nan')):.1f} |")
    nat_div = next((v for k, v in r["diversity"].items() if k.startswith("natural")), None)
    if nat_div:
        add(f"| _natural {pos_name}_ | | | {nat_div['final_compression']:.1f} | {nat_div['cot_compression']:.1f} |")
    for key, cov in r.get("coverage", {}).items():
        if cov and key in key_list[:2]:
            add("", f"Naturalness and coverage ({key}; matching on the prompt; prompt-only AUC near 0.5 means the matching worked):", "", "| generator | precision | recall | matched discriminator AUC | prompt-only AUC |", "|---|---|---|---|---|")
            for g, d in cov.items():
                add(f"| {g} | {d['precision']:.2f} | {d['recall']:.2f} | {d['matched']['auc']:.3f} | {d['matched']['prompt_only_auc']:.3f} |")
    for key in key_list:
        f = r["features"].get(key)
        if not f:
            continue
        for sl in slices:
            if sl not in f:
                continue
            budgets = [b for b in ("8", "16", "64", "256") if b in f[sl]]
            arms = sorted(f[sl][budgets[0]]) if budgets else []
            add("", f"**{key}, slice {sl}**: TPR at 1% FPR (95% interval; `*` = difference from natural excludes 0, shown as +delta)", "", "| arm | " + " | ".join(f"b={b}" for b in budgets) + " |", "|---|" + "---|" * len(budgets))
            order = ["natural"] + [a for a in arms if a != "natural"]
            for a in order:
                cells = []
                for b in budgets:
                    st = f[sl][b].get(a)
                    cells.append((ci(st) + dlt(st)) if st else "-")
                add(f"| {a} | " + " | ".join(cells) + " |")
    ctl = r.get("controls", {})
    for key, c in ctl.items():
        add("", f"**Projection controls ({key})**: probe trained after projecting out a direction, AUROC / TPR at 1% FPR (the unprojected rows are in the tables above)", "",
            "| projected out | " + " | ".join(a for a in next(iter(c.values()))) + " |",
            "|---|" + "---|" * len(next(iter(c.values()))))
        for proj, arms in c.items():
            add(f"| {proj} | " + " | ".join(f"{s['auroc']['mean']:.2f} / {s['tpr@0.01']['mean']:.2f}" for s in arms.values()) + " |")


def syco_sweep():
    t = load("syco_sweep_table.json")
    if not t:
        add("_no sweep_")
        return
    base = t["base"]
    ok = [(c, s) for c, s in t.items() if s.get("family") == "dom" and s["incomplete_rate"] <= base["incomplete_rate"] + 0.15 and s["rep4"] <= base["rep4"] + 0.05]
    ok.sort(key=lambda cs: -cs[1]["pos_rate"])
    add(f"Baseline sycophancy rate on 64 low-sycophancy items: {base['pos_rate']:.2f} (unparsed {base['incomplete_rate']:.2f}).", "", "| condition | sycophantic share | unparsed | rep4 |", "|---|---|---|---|")
    for c, s in ok[:8]:
        add(f"| {c} | {s['pos_rate']:.2f} | {s['incomplete_rate']:.2f} | {s['rep4']:.3f} |")
    rnd = [s["pos_rate"] for s in t.values() if s.get("family") == "random"]
    add("", "Random-direction controls: " + (", ".join(f"{v:.2f}" for v in rnd) or "none"))


def laser():
    r = load("laser_warmstart.json")
    if not r:
        add("_no LASER output_")
        return
    add(f"Pool of benign WildChat traffic: {r['pool']['n_pool']} rows, {r['pool']['pool_pos_rate']:.3f} natural refusals; evaluated on {r['pool']['n_eval']} held-out rows ({r['pool']['eval_pos']} positives).", "")
    for key, arms in r["keys"].items():
        pts = sorted({p["n_labels"] for rows in arms.values() for p in rows})
        sel = [p for p in pts if p in (32, 64, 128, 256) or p == pts[-1]][:4]
        add(f"**{key}**: natural positives found / TPR at 1% FPR after N natural labels (extras are free)", "", "| arm | " + " | ".join(f"N={n}" for n in sel) + " |", "|---|" + "---|" * len(sel))
        for a, rows in sorted(arms.items()):
            cells = []
            for n in sel:
                p = next((x for x in rows if x["n_labels"] == n), None)
                cells.append("-" if not p else f"{p['n_pos']:.0f} / {p['tpr@0.01']['mean']:.2f}" if "tpr@0.01" in p else f"{p['n_pos']:.0f} / -")
            add(f"| {a} | " + " | ".join(cells) + " |")
        add("")


def wild_probes():
    r = load("probes_wildchat.json")
    if not r:
        add("_no WildChat probes_")
        return
    add(f"Held-out WildChat prompts: {r['n_eval']} rows, {r['eval_pos']} natural refusals ({r['eval_pos_rate']:.3f}). Reference arm: natural WildChat-only training data.", "")
    for key, f in r["features"].items():
        f = f["wildchat_heldout"]
        budgets = list(f)
        arms = sorted(f[budgets[0]])
        add(f"**{key}**: TPR at 1% FPR", "", "| arm | " + " | ".join(f"b={b}" for b in budgets) + " |", "|---|" + "---|" * len(budgets))
        for a in ["natural_wild", "natural_any"] + [x for x in arms if x not in ("natural_wild", "natural_any")]:
            add(f"| {a} | " + " | ".join((ci(f[b][a]) + dlt(f[b][a])) if a in f[b] else "-" for b in budgets) + " |")
        add("")


def reach():
    t = load("reach_table.json")
    if not t:
        add("_no reachability output_")
        return
    add("Median per-condition values over steered CoT tokens (or prompt tokens for prompt-only conditions). `recon` = SAE reconstruction error (relative MSE), `nn` = distance to the nearest natural activation relative to the token norm. Natural floor in the last row of each group.", "", "| group | n | recon unsteered | recon steered | nn unsteered | nn steered | perturbation / norm |", "|---|---|---|---|---|---|---|")
    groups = {}
    for e in t:
        g = "dense direction" if e["family"] in ("dom_all", "dom_prompt", "random_dir") else ("bundle" if e["family"] in ("bundle", "valbundle", "random_bundles") else "single latent")
        groups.setdefault(g, []).append(e)
    for g, es in groups.items():
        m = lambda k: np.median([e[k] for e in es])  # noqa: E731
        add(f"| {g} | {len(es)} | {m('rel_mse_unsteered'):.3f} | {m('rel_mse_steered'):.3f} | {m('rel_nn_unsteered'):.3f} | {m('rel_nn_steered'):.3f} | {np.median([e['vec_norm'] / e['act_norm'] for e in es]):.2f} |")
    floor = np.median([e["natural_floor"]["rel_mse"] for e in t])
    add(f"| natural floor | | {floor:.3f} | | {np.median([e['natural_floor']['rel_nn'] for e in t]):.3f} | | |")
    pairs = [(e["rel_mse_steered"] - e["rel_mse_unsteered"], e["stats"]["pos_rate"]) for e in t if e.get("stats") and e["source"] == "validate" and e["family"] == "attr"]
    if len(pairs) >= 5:
        x, y = np.array(pairs).T
        add("", f"Single latents: correlation between the increase in reconstruction error and the refusal yield r = {np.corrcoef(x, y)[0, 1]:.2f} (n = {len(x)}).")


def timing():
    t = load("timing_table.json")
    if not t:
        add("_no timing output_")
        return
    by = {}
    for c, s in t.items():
        by.setdefault(s["kind"], {})[s["positions"]] = s
    add("Refusal yield by steered position set (HF, 32 prompts per cell; incomplete in parentheses):", "", "| steering | prompt only | CoT only | answer only |", "|---|---|---|---|")
    for k, d in by.items():
        add(f"| {k} | " + " | ".join((f"{d[p]['pos_rate']:.2f} ({d[p]['incomplete_rate']:.2f})" if p in d else "-") for p in ("prompt", "cot", "final")) + " |")


add("# steer_v1 morning summary", "", f"Source: `{POOL}`. Intervals are 95% prompt-clustered bootstrap; `*` marks a paired difference from the natural-trained arm whose interval excludes 0.")
section("Stage timings and problems", timings)
section("Refusal: latent search (wider, layers 11/15/19/23)", lambda: latent_search("", "harmless prompts"))
section("Refusal: synthetic data, naturalness, probes", lambda: refusal_generators_and_probes("", "probes_report.json", ["L15_cot_first32", "L15_cot_first8", "L15_cot_mean"], ["all", "harmful_prompts", "harmless_prompts"], "refusals"))
section("Sycophancy: steering sweep with dense directions", syco_sweep)
section("Sycophancy: attributed latents", lambda: latent_search("syco_", "low-sycophancy items"))
section("Sycophancy: synthetic data and probes", lambda: refusal_generators_and_probes("syco_", "syco_probes_report.json", ["L15_final_mean", "L15_cot_first32", "L15_prompt_last"], ["all"], "sycophantic answers"))
section("LASER with a synthetic warm start (benign WildChat traffic)", laser)
section("Probes on held-out WildChat refusals", wild_probes)
section("Reachability of steered activations", reach)
section("Steering timing for latents (HF masks)", timing)
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT, f"({len(lines)} lines)")
