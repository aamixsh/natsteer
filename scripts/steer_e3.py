"""E3 (naturalness-constrained generation) and E2 at scale, on the refusal testbed.

Reads data/pools/steer_v2 (symlinked into steer_e3), writes data/pools/steer_e3 (smoke: steer_e3_smoke).
Every stage skips if its output exists. Stages:

  init           symlink the steer_v2 inputs
  e3_floor       HF: per-token SAE reconstruction error of natural CoT tokens at layers 11, 15, 19 (the cap)
  reach2         HF (E2): reconstruction error and nearest-natural-neighbor distance of steered activations for
                 every refusal and sycophancy condition validated in steer_v2 (working and failing latents,
                 bundles), next to its yield
  e3_gen         vLLM: on 1,500 harmless prompts and on the WildChat prompts the model naturally refuses:
                 K = 8 samples per prompt from fixed steering (dense direction, the top validated latents) and
                 from the prompt ("politely decline"), for best-of-K selection (E3b); and capped steering (E3a:
                 per-token strength under the natural 90th-percentile reconstruction error), 2 samples per prompt,
                 at the usual and at a higher strength
  e3_features    HF: clean pooled features of every generated row
  e3_select      CPU (E3b): per prompt, keep the verified positive that a natural-vs-synthetic discriminator finds
                 most natural; two discriminators: one trained with 128 labeled natural refusals, one label-free
                 (2,000 unlabeled natural samples); baseline = a random verified positive of the same K
  e3_probes      CPU: probes on the refusal test set (harmless-prompt arms) and on held-out WildChat refusals
                 (matched-prompt arms), 10 seeds, prompt-clustered intervals, same protocol as steer_v2
  e2_usefulness  CPU (E2): per validated latent, how useful its data is for probes; correlate yield, distance
                 and usefulness
  summary_e3     SUMMARY.md next to the outputs
  report_e3      copy the report JSONs to data/runs/steer_e3/<stamp>
Usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/steer_e3.py --stage e3_gen [--smoke]
"""

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import steer_v1 as sv  # noqa: E402
import steer_v1_extra as sx  # noqa: E402
import steer_v2 as v2  # noqa: E402
from natsteer.config import run_dir  # noqa: E402
from natsteer.refusal import read_jsonl  # noqa: E402

log = sv.log
POOLS = sv.POOLS
FLOOR_LAYERS = (11, 15, 19)
E3_POOLINGS = ("eop_mean", "cot_first16", "cot_first32", "cot_mean")


class E3Cfg(v2.V2Cfg):
    def __init__(self, smoke):
        super().__init__(smoke)
        self.src = POOLS / ("steer_v2_smoke" if smoke else "steer_v2")
        self.dir = POOLS / ("steer_e3_smoke" if smoke else "steer_e3")
        self.n_prompts = 16 if smoke else 1500
        self.k = 2 if smoke else 8
        self.cap_samples = 1 if smoke else 2
        self.top_latents = 3 if smoke else 8
        self.e3_tokens = 384 if smoke else 1536
        self.cap_quantile = 0.9
        self.floor_rows = 40 if smoke else 800
        self.floor_tokens = 2000 if smoke else 60000
        self.reach_rows = 4 if smoke else 24
        self.reach_bank_tokens = 3000 if smoke else 60000
        self.reach_nn_queries = 400 if smoke else 4096
        self.disc_label_pos = 16 if smoke else 128
        self.disc_free = 200 if smoke else 2000
        self.disc_syn = 200 if smoke else 3000
        self.e3_budgets = (16, 64) if smoke else (16, 64, 256)
        self.use_budgets = (16,) if smoke else (16, 32, 64)


# ------------------------------------------------------------------ setup


def stage_init(cfg, args):
    cfg.dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(cfg.src.iterdir()):
        if p.is_file() and p.name not in ("SUMMARY.md", "timings.log") and not (cfg.dir / p.name).exists():
            (cfg.dir / p.name).symlink_to(p.resolve())
            n += 1
    cfg.path("init.done").write_text("ok")
    log(f"init: linked {n} files from {cfg.src}")


def stage_e3_floor(cfg, args):
    """Quantiles of the per-token relative reconstruction error of natural CoT tokens (unsteered)."""
    from natsteer.model import load_model
    from natsteer.sae import load_andyrdt_sae

    rng = np.random.default_rng(0)
    nat = [r for r in sv.load_natural(cfg) if r.get("final") and r["split"] == "train"]
    pick = [nat[i] for i in rng.choice(len(nat), min(len(nat), cfg.floor_rows), replace=False)]
    model, tok = load_model("bf16")
    out = {}
    for L in FLOOR_LAYERS:
        sae = load_andyrdt_sae(L, 64, device=model.device)
        x, _ = sx.collect_tokens(model, tok, pick, [{"layer": L, "vec": None}], "all", cfg.floor_tokens, rng, steered=False)
        e = sx.recon_rel_mse(sae, x).cpu().numpy()
        out[str(L)] = {"n_tokens": int(len(e)), **{f"q{int(q * 100)}": float(np.quantile(e, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}, "mean": float(e.mean())}
        log(f"L{L}: natural CoT reconstruction error {out[str(L)]}")
        del sae
    json.dump(out, open(cfg.path("e3_floor.json"), "w"), indent=1)


# ------------------------------------------------------------------ E2 at scale


def _reach_conditions(cfg, task):
    import torch

    att = torch.load(cfg.path("attribution_wide.pt" if task == "refusal" else "syco_attribution_wide.pt"))
    conds = v2.validated_conds(cfg, att, task)
    table = json.load(open(cfg.path("latents_validated2_table.json" if task == "refusal" else "syco_latents_validated2_table.json")))
    rows = read_jsonl(cfg.path("latents_validated2.jsonl" if task == "refusal" else "syco_latents_validated2.jsonl"))
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    if task == "refusal":
        nat = [r for r in sv.load_natural(cfg) if r.get("final") and r["split"] == "train"]
    else:
        nat = [r for r in sv.syco_natural_rows(cfg) if r.get("final") and r["split"] == "train"]
    if cfg.smoke:
        conds = conds[:3]
    return conds, table, by, nat


def stage_reach2(cfg, args):
    from natsteer.model import load_model
    from natsteer.sae import load_andyrdt_sae

    model, tok = load_model("bf16")
    rng = np.random.default_rng(0)
    out = []
    saes = {}
    t0 = time.time()
    for task in ("refusal", "syco"):
        conds, table, by, nat = _reach_conditions(cfg, task)
        nat_pick = [nat[i] for i in rng.choice(len(nat), min(len(nat), 800), replace=False)]
        banks, floors = {}, {}
        base = table["base"]
        for k, c in enumerate(conds):
            rows = [r for r in by.get(c["name"], []) if r.get("thinking")]
            if not rows:
                continue
            specs = c["specs"]
            layer = max(s["layer"] for s in specs)
            if layer not in saes:
                saes[layer] = load_andyrdt_sae(layer, 64, device=model.device)
            if layer not in banks:
                x, _ = sx.collect_tokens(model, tok, nat_pick[: len(nat_pick) * 3 // 4], [{"layer": layer, "vec": None}], "all", cfg.reach_bank_tokens, rng, steered=False)
                held, _ = sx.collect_tokens(model, tok, nat_pick[len(nat_pick) * 3 // 4 :], [{"layer": layer, "vec": None}], "all", cfg.reach_nn_queries, rng, steered=False)
                banks[layer] = x
                floors[layer] = {"rel_mse": float(sx.recon_rel_mse(saes[layer], held).mean()), "rel_nn": float(sx.nn_rel_dist(held, x).median())}
            pick = [rows[i] for i in rng.choice(len(rows), min(len(rows), cfg.reach_rows), replace=False)]
            xs, _ = sx.collect_tokens(model, tok, pick, specs, "all", cfg.reach_nn_queries * 2, rng, steered=True)
            xu, _ = sx.collect_tokens(model, tok, pick, specs, "all", cfg.reach_nn_queries * 2, rng, steered=False)
            st = table.get(c["name"], {})
            e = {"task": task, "name": c["name"], "family": c.get("family"), "layer": layer, "latents": c.get("latents"), "lam": c.get("lam"),
                 "pos_rate": st.get("pos_rate"), "incomplete_rate": st.get("incomplete_rate"), "base_rate": base["pos_rate"],
                 "rel_mse_steered": float(sx.recon_rel_mse(saes[layer], xs).mean()), "rel_mse_unsteered": float(sx.recon_rel_mse(saes[layer], xu).mean()),
                 "rel_nn_steered": float(sx.nn_rel_dist(xs[: cfg.reach_nn_queries], banks[layer]).median()), "rel_nn_unsteered": float(sx.nn_rel_dist(xu[: cfg.reach_nn_queries], banks[layer]).median()),
                 "vec_norm": float(sum(s["vec"].float().norm() for s in specs)), "act_norm": float(xu.norm(dim=-1).median()), "natural_floor": floors[layer]}
            out.append(e)
            log(f"{task} {k + 1}/{len(conds)} {c['name']}: yield {e['pos_rate']}, recon {e['rel_mse_unsteered']:.3f}->{e['rel_mse_steered']:.3f}, nn {e['rel_nn_unsteered']:.3f}->{e['rel_nn_steered']:.3f} ({time.time() - t0:.0f}s)")
    json.dump(out, open(cfg.path("reach2_table.json"), "w"), indent=1)


# ------------------------------------------------------------------ E3 generation


def e3_prompt_sets(cfg):
    _, _, data = sv.split_prompts(cfg)
    return {"h": data[: cfg.n_prompts], "m": v2.matched_prompts(cfg)}


def e3_conditions(cfg, att, prefix, floor):
    """(K-sample conditions, capped conditions) for one prompt set. Latent conditions are partitioned over the
    prompts (one latent per prompt group), as in the many-latent generators."""
    import torch

    D = torch.load(cfg.path("directions.pt"))["dirs"]
    sweep = json.load(open(cfg.path("sweep_table.json")))
    choices = json.load(open(cfg.path("generator_choices_v1.json")))
    s = sweep[choices["dom_all"]]
    dense = lambda c: [{"layer": s["layer"], "vec": c * D[s["kind"]][s["layer"]], "positions": "all"}]  # noqa: E731
    _, good = v2.good_validated(cfg)
    allc = {c["name"]: c for c in v2.validated_conds(cfg, att)}
    lat = [c for c in good if c in allc][: cfg.top_latents]
    cap = lambda L: floor[str(L)][f"q{int(cfg.cap_quantile * 100)}"]  # noqa: E731

    def capped(specs, mult, tag):
        return [{"layer": sp["layer"], "vec": mult * sp["vec"], "positions": "all", "cap": cap(sp["layer"]), "tag": tag} for sp in specs]

    k_conds = [{"name": f"{prefix}k_dense", "generator": f"{prefix}k_dense", "specs": dense(s["coeff"])}]
    k_conds += [{"name": f"{prefix}k_latents:{c}", "generator": f"{prefix}k_latents", "specs": allc[c]["specs"], "part": i} for i, c in enumerate(lat)]
    k_conds += [{"name": f"{prefix}k_prompted", "generator": f"{prefix}k_prompted", "specs": [], "prompted": True}]
    cap_conds = [{"name": f"{prefix}cap_dense_c{c}", "generator": f"{prefix}cap_dense_c{c}", "specs": capped(dense(c), 1.0, f"{prefix}cap_dense_c{c}")} for c in (s["coeff"], 1.5 * s["coeff"])]
    for mult in (1.0, 1.5):
        cap_conds += [{"name": f"{prefix}cap_latents_x{mult}:{c}", "generator": f"{prefix}cap_latents_x{mult}", "specs": capped(allc[c]["specs"], mult, f"{prefix}cap_latents_x{mult}"), "part": i} for i, c in enumerate(lat)]
    return k_conds, cap_conds, len(lat)


def stage_e3_gen(cfg, args):
    import torch

    from natsteer.vllm_steer import alpha_log, make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.80)  # engine first; leave room for the SAEs in the hooks
    att = torch.load(cfg.path("attribution_wide.pt"))
    floor = json.load(open(cfg.path("e3_floor.json")))
    path = cfg.path("e3_gen.jsonl")
    for prefix, prompts in ((p + "_", ps) for p, ps in e3_prompt_sets(cfg).items()):
        k_conds, cap_conds, n_lat = e3_conditions(cfg, att, prefix, floor)
        parts = sv.partition(prompts, max(1, n_lat))
        for group, n in ((k_conds, cfg.k), (cap_conds, cfg.cap_samples)):
            for c in group:
                if "part" in c:
                    c["prompts"] = parts[c["part"]]
                if c.get("prompted"):
                    c["prompts"] = [p | {"gen_user": p["user"] + sv.PROMPTED_REFUSAL} for p in prompts]
            log(f"{prefix}: {len(group)} conditions x {n} samples on {len(prompts)} prompts")
            sv.run_conditions(llm, group, prompts, n, cfg.e3_tokens, path, sv.refusal_labeler)
    logs = llm.apply_model(alpha_log)[0]
    json.dump({k: {"mean_alpha": v[0] / max(v[1], 1), "tokens": v[1]} for k, v in logs.items()}, open(cfg.path("e3_alpha.json"), "w"), indent=1)
    log("mean per-token alpha under the cap:", {k: round(v[0] / max(v[1], 1), 3) for k, v in logs.items()})
    cfg.path("e3_gen.done").write_text("ok")


def stage_e3_features(cfg, args):
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    sv.extract_to(cfg, model, tok, read_jsonl(cfg.path("e3_gen.jsonl")), "e3_gen", sv.FEAT_LAYERS, E3_POOLINGS)


# ------------------------------------------------------------------ E3b selection


def stage_e3_select(cfg, args):
    """Per prompt and generator: a random verified positive (baseline) and the most natural-looking verified
    positive under two discriminators (labeled: 128 natural refusals; label-free: unlabeled natural samples).
    Discriminators are trained on earlier synthetic positives (steer_v1/v2), never on the rows they score."""
    from natsteer.probes import make_probe

    key = "L15_cot_first32"
    nat_rows, nat_f = sv.load_feats(cfg, "natural", sv.load_natural(cfg))
    _, _, prev_rows, prev_f = v2.refusal_data(cfg, ["synthetic", "synthetic2"])
    rows, f = sv.load_feats(cfg, "e3_gen", "e3_gen.jsonl")
    rng = np.random.default_rng(0)
    yn = np.array([r["label"] for r in nat_rows])
    tr = np.array([r["split"] == "train" for r in nat_rows])
    Xn = nat_f[key].astype(np.float32)
    okn = ~np.isnan(Xn).any(1)
    Xp = prev_f[key].astype(np.float32)
    prev_pos = np.flatnonzero((np.array([r["label"] for r in prev_rows]) == 1) & ~np.isnan(Xp).any(1) & (np.array([r["generator"] for r in prev_rows]) != "random_dir"))
    nat_pos = np.flatnonzero(tr & okn & (yn == 1))
    nat_any = np.flatnonzero(tr & okn)
    disc = {}
    for name, nat_idx, n_syn in (("label", rng.choice(nat_pos, min(len(nat_pos), cfg.disc_label_pos), replace=False), 4 * cfg.disc_label_pos), ("free", rng.choice(nat_any, min(len(nat_any), cfg.disc_free), replace=False), cfg.disc_syn)):
        syn_idx = rng.choice(prev_pos, min(len(prev_pos), n_syn), replace=False)
        X = np.concatenate([Xp[syn_idx], Xn[nat_idx]])
        y = np.r_[np.ones(len(syn_idx)), np.zeros(len(nat_idx))].astype(int)
        disc[name] = make_probe().fit(X, y)
        log(f"discriminator {name}: {len(syn_idx)} earlier synthetic positives vs {len(nat_idx)} natural rows")
    Xe = f[key].astype(np.float32)
    ok = ~np.isnan(Xe).any(1)
    lab = np.array([r["label"] for r in rows])
    score = {name: np.full(len(rows), np.nan) for name in disc}
    for name, d in disc.items():
        score[name][ok] = d.decision_function(Xe[ok])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if lab[i] == 1 and ok[i]:
            groups[(r["generator"], r["prompt_id"])].append(i)
    sel = defaultdict(list)
    for (g, pid), idx in sorted(groups.items()):
        if "_k_" in g:
            sel[g + "_rand"].append(int(rng.choice(idx)))
            for name in disc:
                sel[g + f"_best_{name}"].append(int(min(idx, key=lambda i: score[name][i])))
        else:  # capped steering: the first verified positive of the prompt's samples
            sel[g].append(int(idx[0]))
    stats = {}
    for arm, idx in sel.items():
        stats[arm] = {"n": len(idx), **{f"mean_score_{n}": float(np.mean(score[n][idx])) for n in disc}}
    gen = np.array([r["generator"] for r in rows])
    yields = {g: {"samples": int((gen == g).sum()), "verified": int(((gen == g) & (lab == 1)).sum()), "cut_off": int(((gen == g) & np.array([x is None for x in lab])).sum())} for g in sorted(set(gen))}
    json.dump({"arms": {a: [rows[i]["id"] for i in idx] for a, idx in sel.items()}, "stats": stats, "yields": yields}, open(cfg.path("e3_select.json"), "w"), indent=1)
    for a, s in sorted(stats.items()):
        log(f"{a}: {s}")


# ------------------------------------------------------------------ probes


def stage_e3_probes(cfg, args):
    from natsteer.probe_eval import run_arm_grid

    nat_rows, nat_f = sv.load_feats(cfg, "natural", sv.load_natural(cfg))
    rows, f = sv.load_feats(cfg, "e3_gen", "e3_gen.jsonl")
    sel = json.load(open(cfg.path("e3_select.json")))["arms"]
    idx_of = {r["id"]: i for i, r in enumerate(rows)}
    pools_all = {a: np.array([idx_of[i] for i in ids], int) for a, ids in sel.items()}
    c = v2.refusal_context(cfg, nat_rows)
    yn, pid = c["yn"], c["pid"]
    held = np.array([sx.wild_heldout(p) for p in pid])
    wild = np.array([r["source"] == "wildchat" for r in nat_rows])
    report = {"arms": {a: int(len(p)) for a, p in pools_all.items()}, "test_set": {}, "wildchat": {}}
    for key in ("L15_cot_first16", "L15_cot_first32", "L15_cot_mean"):
        Xn, Xs = nat_f[key].astype(np.float32), f[key].astype(np.float32)
        okn, oks = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
        pools = {a: p[oks[p]] for a, p in pools_all.items()}
        # harmless-prompt arms, refusal test set (same protocol as steer_v2 probes2)
        tr, ev, neg = (np.flatnonzero(c[m] & okn) for m in ("train_mask", "eval_mask", "neg_mask"))
        slices = {k: v for k, v in c["slices"](ev).items() if len(v) >= 20}
        h = {a.removeprefix("h_"): p for a, p in pools.items() if a.startswith("h_")}
        arms = v2.make_arms(Xn, yn, tr, neg, Xs, h, mix_names=tuple(a for a in h if "best" in a or "rand" in a))
        report["test_set"][key] = run_arm_grid(arms, Xn[ev], yn[ev], pid[ev], {k: np.isin(ev, v) for k, v in slices.items()}, cfg.e3_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs)
        log(f"{key}: test-set probes done ({len(arms)} arms)")
        # matched-prompt arms, held-out WildChat (same protocol as steer_v2 probes_matched)
        ev_w = np.flatnonzero(wild & held & okn)
        wild_tr = np.flatnonzero(wild & ~held & okn)
        m = {a.removeprefix("m_"): p for a, p in pools.items() if a.startswith("m_")}
        arms = v2.make_arms(Xn, yn, wild_tr, wild_tr[yn[wild_tr] == 0], Xs, m)
        report["wildchat"][key] = run_arm_grid(arms, Xn[ev_w], yn[ev_w], pid[ev_w], {"wildchat_heldout": np.ones(len(ev_w), bool)}, cfg.wild_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs)
        log(f"{key}: WildChat probes done ({len(arms)} arms)")
    json.dump(report, open(cfg.path("e3_probes_report.json"), "w"), indent=1)


def stage_e2_usefulness(cfg, args):
    """Per validated refusal latent: probe quality of its own synthetic positives (steer_v2 many-latent data),
    against its yield and its distance from natural activations."""
    from scipy.stats import spearmanr

    from natsteer.probe_eval import run_arm_grid

    nat_rows, nat_f, syn_rows, syn_f = v2.refusal_data(cfg, ["synthetic2"])
    c = v2.refusal_context(cfg, nat_rows)
    yn, pid = c["yn"], c["pid"]
    key = "L15_cot_first32"
    Xn, Xs = nat_f[key].astype(np.float32), syn_f[key].astype(np.float32)
    okn, oks = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
    tr, ev, neg = (np.flatnonzero(c[m] & okn) for m in ("train_mask", "eval_mask", "neg_mask"))
    cond = np.array([r["condition"] for r in syn_rows])
    ys = np.array([r["label"] for r in syn_rows])
    pools = {cn.split(":", 1)[1]: np.flatnonzero((cond == cn) & (ys == 1) & oks) for cn in sorted(set(cond)) if cn.startswith("many_mix:")}
    arms = v2.make_arms(Xn, yn, tr, neg, Xs, pools)
    res = run_arm_grid(arms, Xn[ev], yn[ev], pid[ev], {"all": np.ones(len(ev), bool)}, cfg.use_budgets, cfg.seeds, n_boot=max(20, cfg.n_boot // 4), n_jobs=cfg.n_jobs)["all"]
    reach = {e["name"]: e for e in json.load(open(cfg.path("reach2_table.json"))) if e["task"] == "refusal"}
    b = str(max(int(x) for x in res if res[x].get("natural")))
    rows = []
    for name in pools:
        st = res[b].get("synthetic_" + name)
        if st is None or name not in reach:
            continue
        e = reach[name]
        rows.append({"latent": name, "n_pos": int(len(pools[name])), "tpr": st["tpr@0.01"]["mean"], "auroc": st["auroc"]["mean"], "yield": e["pos_rate"], "recon_increase": e["rel_mse_steered"] - e["rel_mse_unsteered"], "nn_steered": e["rel_nn_steered"]})
    corr = {}
    for x in ("yield", "recon_increase", "nn_steered"):
        for y in ("tpr", "auroc"):
            if len(rows) >= 5:
                r = spearmanr([q[x] for q in rows], [q[y] for q in rows])
                corr[f"{x}~{y}"] = {"spearman": float(r.statistic), "p": float(r.pvalue), "n": len(rows)}
    allr = json.load(open(cfg.path("reach2_table.json")))
    for task in ("refusal", "syco"):
        t = [e for e in allr if e["task"] == task and e["pos_rate"] is not None and e["family"] == "attr"]
        if len(t) >= 5:
            for x in ("rel_mse_steered", "rel_nn_steered"):
                r = spearmanr([e[x] - (e["rel_mse_unsteered"] if x == "rel_mse_steered" else e["rel_nn_unsteered"]) for e in t], [e["pos_rate"] for e in t])
                corr[f"{task}:{x}_increase~yield"] = {"spearman": float(r.statistic), "p": float(r.pvalue), "n": len(t)}
    json.dump({"budget": b, "natural": res[b]["natural"], "latents": rows, "correlations": corr}, open(cfg.path("e2_usefulness.json"), "w"), indent=1)
    log("correlations:", json.dumps(corr))


# ------------------------------------------------------------------ summary


def stage_summary_e3(cfg, args):
    L = []
    add = L.extend
    load = lambda n: json.load(open(cfg.path(n))) if cfg.path(n).exists() else None  # noqa: E731

    def ci(st):
        m = st["tpr@0.01"]
        d = st.get("delta_tpr@0.01")
        return f"{m['mean']:.2f} [{m['lo']:.2f},{m['hi']:.2f}]" + (f" {d['mean']:+.2f}{'*' if d['lo'] > 0 or d['hi'] < 0 else ''}" if d else "")

    def grid(f, title):
        budgets = list(f)
        arms = sorted({a for b in budgets for a in f[b]}, key=lambda a: (a != "natural", a))
        add(["", title, "", "| arm | " + " | ".join(f"b={b}" for b in budgets) + " |", "|---|" + "---|" * len(budgets)])
        for a in arms:
            add([f"| {a} | " + " | ".join(ci(f[b][a]) if a in f[b] else "-" for b in budgets) + " |"])

    add(["# steer_e3 summary (E3 naturalness-constrained generation, E2 at scale)", "", f"Source: `{cfg.dir}`. TPR at 1% FPR with 95% prompt-clustered intervals; `*` = paired difference from natural excludes 0.", ""])
    tl = cfg.path("timings.log")
    if tl.exists():
        add(["## Stages", "", "```", tl.read_text().strip(), "```"])
    fl = load("e3_floor.json")
    if fl:
        add(["", "## Cap: natural per-token reconstruction error (CoT tokens)", "", "| layer | q50 | q90 | q99 |", "|---|---|---|---|"] + [f"| {k} | {v['q50']:.3f} | {v['q90']:.3f} | {v['q99']:.3f} |" for k, v in fl.items()])
    al = load("e3_alpha.json")
    if al:
        add(["", "Mean per-token strength kept under the cap: " + ", ".join(f"{k} {v['mean_alpha']:.2f}" for k, v in al.items())])
    se = load("e3_select.json")
    if se:
        add(["", "## Generation yields and selection", "", "| generator | samples | verified positives | cut off |", "|---|---|---|---|"] + [f"| {g} | {v['samples']} | {v['verified']} | {v['cut_off']} |" for g, v in se["yields"].items()])
        add(["", "| arm | positives | discriminator score (label) | (label-free) |", "|---|---|---|---|"] + [f"| {a} | {s['n']} | {s['mean_score_label']:.2f} | {s['mean_score_free']:.2f} |" for a, s in sorted(se["stats"].items())])
    pr = load("e3_probes_report.json")
    if pr:
        add(["", "## Probes: harmless-prompt arms on the refusal test set"])
        for key, res in pr["test_set"].items():
            for sl in ("all", "harmless_prompts"):
                if sl in res:
                    grid(res[sl], f"**{key}, slice {sl}**")
        add(["", "## Probes: matched-prompt arms on held-out WildChat refusals"])
        for key, res in pr["wildchat"].items():
            grid(res["wildchat_heldout"], f"**{key}**")
    us = load("e2_usefulness.json")
    if us:
        add(["", "## E2: yield, distance from natural activations, and usefulness of each latent's data", "", f"Usefulness = TPR at 1% FPR of a probe trained only on that latent's positives (first 32 CoT tokens, {us['budget']} labels; natural {us['natural']['tpr@0.01']['mean']:.2f}).", "", "| correlation | Spearman | p | n |", "|---|---|---|---|"] + [f"| {k} | {v['spearman']:.2f} | {v['p']:.3f} | {v['n']} |" for k, v in us["correlations"].items()])
        add(["", "| latent | positives | yield | recon increase | nn distance | TPR | AUROC |", "|---|---|---|---|---|---|---|"] + [f"| {r['latent']} | {r['n_pos']} | {r['yield']:.2f} | {r['recon_increase']:.3f} | {r['nn_steered']:.3f} | {r['tpr']:.2f} | {r['auroc']:.3f} |" for r in sorted(us["latents"], key=lambda r: -r["tpr"])])
    cfg.path("SUMMARY.md").write_text("\n".join(L) + "\n")
    log("wrote", cfg.path("SUMMARY.md"))


def stage_report_e3(cfg, args):
    out = run_dir("steer_e3_smoke" if cfg.smoke else "steer_e3")
    for p in sorted(cfg.dir.glob("*.json")) + [cfg.path("SUMMARY.md")]:
        if p.exists() and not p.is_symlink():
            shutil.copy(p, out / p.name)
    (out / "report.done").write_text("ok")
    log("saved", out)


STAGES = {
    "init": (stage_init, "init.done"),
    "e3_floor": (stage_e3_floor, "e3_floor.json"),
    "reach2": (stage_reach2, "reach2_table.json"),
    "e3_gen": (stage_e3_gen, "e3_gen.done"),
    "e3_features": (stage_e3_features, "e3_gen.feats.npz"),
    "e3_select": (stage_e3_select, "e3_select.json"),
    "e3_probes": (stage_e3_probes, "e3_probes_report.json"),
    "e2_usefulness": (stage_e2_usefulness, "e2_usefulness.json"),
    "summary_e3": (stage_summary_e3, "summary_e3.never"),
    "report_e3": (stage_report_e3, "report_e3.never"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, outputs under data/pools/steer_e3_smoke (reads steer_v2_smoke)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = E3Cfg(args.smoke)
    random.seed(0)
    fn, marker = STAGES[args.stage]
    if args.stage != "init" and not cfg.path("init.done").exists():
        stage_init(cfg, args)
    if cfg.path(marker).exists() and not args.force:
        log(f"{args.stage}: {marker} exists, skipping")
        return
    t0 = time.time()
    fn(cfg, args)
    log(f"stage {args.stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
