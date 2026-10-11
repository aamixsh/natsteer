"""Third pass: why do synthetic positives lose to natural data, and can a wider latent pool change that?
Reads data/pools/steer_v1 (symlinked into steer_v2), writes data/pools/steer_v2 (smoke: steer_v2_smoke).
Every stage skips if its output exists. Stages:

  init            symlink the steer_v1 outputs into steer_v2
  syco_probes2    CPU: sycophancy probes again with a random-direction arm (does the steering vector matter, or
                  only the exact label?), CoT-length-matched synthetic arms, a length-only probe, label budgets
                  up to 2,048
  screen2         vLLM: more latents for refusal: ranks 33-96 on layers 11 and 15 (strengths 5, 8), the top 32
                  on layers 11/15/19 at strength 8, v1 near misses at strength 12
  validate2       vLLM: best condition per latent (v1 and v2 screens merged) on 96 fresh prompts, plus bundles
  generate2       vLLM: synthetic data from many validated latents (one per prompt group, random 3-latent
                  bundles), a dense direction at several strengths and layers, the best bundle
  matched_gen     vLLM: topic-matched positives. WildChat prompts the model naturally refuses, steered toward
                  refusal and sampled unsteered, 12 samples each, so natural and synthetic positives share prompts
  features2       HF: clean pooled features of the new synthetic and matched rows
  probes2         CPU: standard probe grid (10 seeds, clustered intervals) with the new generators
  probes2_extra   CPU: union of generators, naturalness-filtered synthetic positives (cross-fitted discriminator),
                  multi-layer (11+15+19) features
  probes_matched  CPU: probes on held-out WildChat refusals: natural vs prompt-matched synthetic positives
  summary2        SUMMARY.md next to the outputs
  report2         copy the report JSONs to data/runs/steer_v2/<stamp>
Usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/steer_v2.py --stage screen2 [--smoke]
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
from natsteer.config import run_dir  # noqa: E402
from natsteer.refusal import read_jsonl, write_jsonl  # noqa: E402
from steer_v1_extra import wild_heldout  # noqa: E402

log = sv.log
POOLS = sv.POOLS


class V2Cfg(sv.Cfg):
    def __init__(self, smoke):
        super().__init__(smoke)
        self.v1 = POOLS / ("steer_v1_smoke" if smoke else "steer_v1")
        self.dir = POOLS / ("steer_v2_smoke" if smoke else "steer_v2")
        self.new_layers = (11,) if smoke else (11, 15)
        self.new_ranks = (2, 4) if smoke else (32, 128)
        self.new_lams = (5.0,) if smoke else (5.0, 8.0)
        self.hi_layers = (11,) if smoke else (11, 15, 19)
        self.hi_top = 2 if smoke else 32
        self.hi_lams = (8.0,)
        self.near_min = -1.0 if smoke else 0.05
        self.near_lams = (12.0,)
        self.validate_cap = 4 if smoke else 40
        self.max_many = 6 if smoke else 24
        self.n_many_bundles = 4 if smoke else 60
        self.dense_multi_k = 3 if smoke else 4
        self.match_samples = 3 if smoke else 12
        self.match_tokens = 512 if smoke else 1536
        self.syco_budgets = (16, 64) if smoke else (16, 64, 256, 1024, 2048)
        self.v2_budgets = (16, 64) if smoke else (16, 64, 256)
        self.wild_budgets = (8, 16) if smoke else (8, 16, 32, 64, 128)
        # sycophancy counterpart of the wider latent search
        self.syco_new_layers = (11,) if smoke else (11, 15, 19)
        self.syco_new_ranks = (2, 4) if smoke else (24, 72)
        self.syco_new_lams = (5.0,) if smoke else (5.0, 8.0)
        self.syco_hi_top = 2 if smoke else 24
        self.syco_near_min = -1.0 if smoke else 0.10
        self.syco_validate_cap = 4 if smoke else 40
        self.syco_n_bundles = 4 if smoke else 40
        self.syco_dense_min_gain = 0.0 if smoke else 0.25
        self.layer_extra = (7, 23)
        self.layer_poolings = ("cot_first32", "cot_mean")


# ------------------------------------------------------------------ shared helpers


def stage_init(cfg, args):
    cfg.dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(cfg.v1.iterdir()):
        if p.is_file() and p.name not in ("SUMMARY.md", "timings.log") and not (cfg.dir / p.name).exists():
            (cfg.dir / p.name).symlink_to(p.resolve())
            n += 1
    cfg.path("init.done").write_text("ok")
    log(f"init: linked {n} files from {cfg.v1}")


def ranked(att, L):
    """Latent ids of layer L by delta-attribution, best first (dead latents last)."""
    import torch

    r = att[L]
    score = r["delta_attr"].clone()
    score[r["act_freq"] <= 1e-5] = -1e9
    return torch.argsort(score, descending=True).tolist()


def attr_cond(att, L, i, lam):
    return {"name": f"attr_L{L}_{i}_x{lam}", "family": "attr", "layer": L, "latents": [i], "lam": lam, "positions": "all", "specs": [{"layer": L, "vec": sv.latent_vec(L, i, lam, att[L]["act_mean_active"]), "positions": "all"}]}


TASK_FILES = {"refusal": ("latents_screen_table.json", "screen2_table.json"), "syco": ("syco_latents_screen_table.json", "syco_screen2_table.json")}


def merged_hits(cfg, min_gain, task="refusal"):
    """One entry per latent (best positive rate over the v1 and v2 screens) that works with generation intact."""
    v1 = json.load(open(cfg.path(TASK_FILES[task][0])))
    s2 = json.load(open(cfg.path(TASK_FILES[task][1])))
    base = s2["base"]
    best = {}
    for tbl in (v1, s2):
        for c, s in tbl.items():
            if s.get("family") != "attr" or not sv.acceptable(s, base) or s["pos_rate"] < base["pos_rate"] + min_gain:
                continue
            k = (s["layer"], s["latents"][0])
            if k not in best or s["pos_rate"] > best[k][1]["pos_rate"]:
                best[k] = (c, s)
    return {"base": base} | dict(best.values())


def validated_conds(cfg, att, task="refusal"):
    gain, cap = (cfg.refusal_gain, cfg.validate_cap) if task == "refusal" else (cfg.syco_gain, cfg.syco_validate_cap)
    return sv.validate_conditions(att, merged_hits(cfg, gain, task), cfg, min_gain=gain, cap=cap)[0]


def good_validated(cfg, task="refusal"):
    t = json.load(open(cfg.path("latents_validated2_table.json" if task == "refusal" else "syco_latents_validated2_table.json")))
    base = t["base"]
    gain = cfg.refusal_gain if task == "refusal" else cfg.syco_gain
    good = {c: s for c, s in t.items() if s.get("family") == "attr" and sv.acceptable(s, base) and s["pos_rate"] >= base["pos_rate"] + gain}
    return t, dict(sorted(good.items(), key=lambda cs: -cs[1]["pos_rate"]))


# ------------------------------------------------------------------ GPU: latent search and generators


def stage_screen2(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)  # engine first
    att = torch.load(cfg.path("attribution_wide.pt"))
    select, _, _ = sv.split_prompts(cfg)
    v1 = json.load(open(cfg.path("latents_screen_table.json")))
    conds = [{"name": "base", "specs": [], "family": "base"}]
    seen = set(v1)

    def add(L, i, lam):
        c = attr_cond(att, L, i, lam)
        if c["name"] not in seen:
            seen.add(c["name"])
            conds.append(c)

    for L in cfg.new_layers:
        for i in ranked(att, L)[cfg.new_ranks[0] : cfg.new_ranks[1]]:
            for lam in cfg.new_lams:
                add(L, i, lam)
    for L in cfg.hi_layers:
        for i in ranked(att, L)[: cfg.hi_top]:
            for lam in cfg.hi_lams:
                add(L, i, lam)
    for s in v1.values():
        if s.get("family") == "attr" and s["pos_rate"] >= cfg.near_min:
            for lam in cfg.near_lams:
                add(s["layer"], s["latents"][0], lam)
    log(f"{len(conds)} screening conditions on {cfg.screen_prompts} prompts")
    rows = sv.run_conditions(llm, conds, select[: cfg.screen_prompts], 1, cfg.screen_tokens, cfg.path("screen2.jsonl"), sv.refusal_labeler, keep_text=False)
    json.dump(sv.condition_table(rows), open(cfg.path("screen2_table.json"), "w"), indent=1)


def stage_validate2(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("attribution_wide.pt"))
    _, val, _ = sv.split_prompts(cfg)
    conds = validated_conds(cfg, att)
    log(f"{len(conds)} validation conditions on {len(val)} fresh prompts")
    rows = sv.run_conditions(llm, [{"name": "base", "specs": [], "family": "base"}] + conds, val, cfg.val_samples, cfg.val_tokens, cfg.path("latents_validated2.jsonl"), sv.refusal_labeler)
    json.dump(sv.condition_table(rows), open(cfg.path("latents_validated2_table.json"), "w"), indent=1)
    _, good = good_validated(cfg)
    by_layer = defaultdict(int)
    for s in good.values():
        by_layer[s["layer"]] += 1
    log(f"validated single latents (>=20% on fresh prompts, generation intact): {len(good)} {dict(by_layer)}")


def stage_generate2(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("attribution_wide.pt"))
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    sweep = json.load(open(cfg.path("sweep_table.json")))
    _, _, data = sv.split_prompts(cfg)
    vt, good = good_validated(cfg)
    conds = {c["name"]: c for c in validated_conds(cfg, att)}
    lat = list(good)[: cfg.max_many]
    log(f"{len(good)} validated latents, using {len(lat)}: {lat[:8]}")
    gens = []
    for c, part in zip(lat, sv.partition(data, max(1, len(lat)))):
        gens.append({"name": f"many_mix:{c}", "generator": "many_mix", "specs": conds[c]["specs"], "prompts": part})
    if len(lat) >= 3:
        rng = random.Random(0)
        for b, part in enumerate(sv.partition(data, cfg.n_many_bundles)):
            by_layer = defaultdict(lambda: 0)
            for c in rng.sample(lat, 3):
                s = vt[c]
                by_layer[s["layer"]] = by_layer[s["layer"]] + rng.uniform(0.5, 1.0) * 0.6 * sv.latent_vec(s["layer"], s["latents"][0], s["lam"], att[s["layer"]]["act_mean_active"])
            gens.append({"name": f"many_bundles:{b}", "generator": "many_bundles", "specs": [{"layer": L, "vec": v, "positions": "all"} for L, v in by_layer.items()], "prompts": part})
    base_s = sweep["base"]
    ok = sorted([(c, s) for c, s in sweep.items() if s.get("family") == "dom" and s["positions"] == "all" and sv.acceptable(s, base_s) and s["refusal_rate"] >= (0.0 if cfg.smoke else 0.8)], key=lambda cs: -cs[1]["refusal_rate"])
    picked, seen = [], set()
    for c, s in ok:
        if (s["kind"], s["layer"]) not in seen:
            seen.add((s["kind"], s["layer"]))
            picked.append((c, s))
    picked = picked[: cfg.dense_multi_k]
    log("dense_multi settings:", [c for c, _ in picked])
    for (c, s), part in zip(picked, sv.partition(data, max(1, len(picked)))):
        gens.append({"name": f"dense_multi:{c}", "generator": "dense_multi", "specs": [{"layer": s["layer"], "vec": s["coeff"] * D[s["kind"]][s["layer"]], "positions": "all"}], "prompts": part})
    bund = sorted([c for c, s in vt.items() if s.get("family") == "bundle" and sv.acceptable(s, vt["base"]) and s["pos_rate"] >= vt["base"]["pos_rate"] + cfg.refusal_gain], key=lambda c: -vt[c]["pos_rate"])
    if bund:
        log("best bundle", bund[0], vt[bund[0]]["pos_rate"])
        gens.append({"name": "valbundle2", "generator": "valbundle2", "specs": conds[bund[0]]["specs"], "prompts": data})
    json.dump({"latents": lat, "dense_multi": [c for c, _ in picked], "valbundle2": bund[:1]}, open(cfg.path("generator_choices_v2.json"), "w"), indent=1)
    sv.run_conditions(llm, gens, data, 1, cfg.gen_tokens, cfg.path("synthetic2.jsonl"), sv.refusal_labeler)
    cfg.path("synthetic2.done").write_text("ok")


def matched_prompts(cfg):
    """WildChat training prompts (not held out) with at least one natural refusal among their samples."""
    by = defaultdict(list)
    for r in sv.load_natural(cfg):
        if r["source"] == "wildchat":
            by[r["prompt_id"]].append(r)
    out = [{"prompt_id": p, "user": rs[0]["user"], "source": "wildchat"} for p, rs in sorted(by.items()) if not wild_heldout(p) and any(r["label"] == 1 for r in rs)]
    if not out:  # smoke runs keep too few WildChat prompts to contain a refusal
        out = [{"prompt_id": p, "user": rs[0]["user"], "source": "wildchat"} for p, rs in sorted(by.items()) if not wild_heldout(p)][:6]
    return out


def stage_matched_gen(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("attribution_wide.pt"))
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    sweep = json.load(open(cfg.path("sweep_table.json")))
    choices = json.load(open(cfg.path("generator_choices_v1.json")))
    prompts = matched_prompts(cfg)
    log(f"{len(prompts)} naturally refused WildChat prompts x {cfg.match_samples} samples")
    conds = [{"name": "matched_base", "generator": "matched_base", "specs": []}]
    if "dom_all" in choices:
        s = sweep[choices["dom_all"]]
        conds.append({"name": "matched_dom_all", "generator": "matched_dom_all", "specs": [{"layer": s["layer"], "vec": s["coeff"] * D[s["kind"]][s["layer"]], "positions": "all"}]})
    if "valbundle" in choices:  # the v1 validated bundle, rebuilt from the v1 screen as in steer_v1.stage_generate
        v1c, _ = sv.validate_conditions(att, json.load(open(cfg.path("latents_screen_table.json"))), cfg, min_gain=cfg.refusal_gain, cap=48 if not cfg.smoke else 3)
        byname = {c["name"]: c for c in v1c}
        if choices["valbundle"] in byname:
            conds.append({"name": "matched_valbundle", "generator": "matched_valbundle", "specs": byname[choices["valbundle"]]["specs"]})
    _, good = good_validated(cfg)
    allc = {c["name"]: c for c in validated_conds(cfg, att)}
    for c in list(good)[:2]:
        conds.append({"name": f"matched_{c}", "generator": "matched_latent", "specs": allc[c]["specs"]})
    conds.append({"name": "matched_prompted", "generator": "matched_prompted", "specs": [], "prompts": [p | {"gen_user": p["user"] + sv.PROMPTED_REFUSAL} for p in prompts]})
    sv.run_conditions(llm, conds, prompts, cfg.match_samples, cfg.match_tokens, cfg.path("matched.jsonl"), sv.refusal_labeler)
    cfg.path("matched.done").write_text("ok")


def stage_features2(cfg, args):
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    for name, src in (("synthetic2", "synthetic2.jsonl"), ("matched", "matched.jsonl")):
        if not cfg.path(f"{name}.feats.npz").exists():
            sv.extract_to(cfg, model, tok, read_jsonl(cfg.path(src)), name, sv.FEAT_LAYERS, sv.FEAT_POOLINGS)


# ------------------------------------------------------------------ CPU: probe helpers


def make_arms(Xn, yn, tr_k, neg_k, Xs, pos_sets, mix_names=()):
    """natural: balanced sample of natural training rows (tr_k). synthetic_<name>: positives from pos_sets[name]
    (indices into Xs), negatives from natural rows neg_k. mix_<name>: half natural, half that."""
    trp, trn = tr_k[yn[tr_k] == 1], tr_k[yn[tr_k] == 0]

    def natural(b, seed):
        rng = np.random.default_rng(seed)
        if min(len(trp), len(trn)) < b // 2:
            return None
        i = np.r_[rng.choice(trp, b // 2, replace=False), rng.choice(trn, b // 2, replace=False)]
        return Xn[i], yn[i]

    def synth(pool, frac):
        def fn(b, seed):
            rng = np.random.default_rng(100 + seed)
            k = int(b * frac / 2)
            if len(pool) < b // 2 - k or len(neg_k) < b // 2 - k or min(len(trp), len(trn)) < k:
                return None
            i_s, i_n = rng.choice(pool, b // 2 - k, replace=False), rng.choice(neg_k, b // 2 - k, replace=False)
            X, y = [Xs[i_s], Xn[i_n]], [np.ones(len(i_s)), np.zeros(len(i_n))]
            if k:
                j = np.r_[rng.choice(trp, k, replace=False), rng.choice(trn, k, replace=False)]
                X.append(Xn[j])
                y.append(yn[j])
            return np.concatenate(X), np.concatenate(y).astype(int)

        return fn

    arms = {"natural": natural}
    for name, pool in pos_sets.items():
        if len(pool) == 0:
            continue
        arms["synthetic_" + name] = synth(pool, 0.0)
        if name in mix_names:
            arms["mix_" + name] = synth(pool, 0.5)
    return arms


def refusal_data(cfg, syn_names):
    """Natural rows and features plus the synthetic rows and features of the listed files (concatenated)."""
    nat_rows, nat_f = sv.load_feats(cfg, "natural", sv.load_natural(cfg))
    rows, feats = [], []
    for name in syn_names:
        r, f = sv.load_feats(cfg, name, name + ".jsonl")
        rows += r
        feats.append(f)
    keys = [k for k in feats[0].files if k != "ids" and all(k in f.files for f in feats)]
    syn_f = {k: np.concatenate([f[k] for f in feats]) for k in keys}
    return nat_rows, nat_f, rows, syn_f


def refusal_context(cfg, nat_rows):
    _, _, data = sv.split_prompts(cfg)
    data_ids = {p["prompt_id"] for p in data}
    split = np.array([r["split"] for r in nat_rows])
    src = np.array([r["source"] for r in nat_rows])
    hl = np.array([r.get("human_label") if r.get("human_label") is not None else -1 for r in nat_rows])
    pid = np.array([r["prompt_id"] for r in nat_rows])
    yn = np.array([r["label"] for r in nat_rows])

    def slices(ev_idx):
        e = np.zeros(len(nat_rows), bool)
        e[ev_idx] = True
        return {"all": ev_idx, "harmful_prompts": np.flatnonzero(e & (np.isin(src, ["test_harmful"]) | ((src == "xstest") & (hl == 1)))), "harmless_prompts": np.flatnonzero(e & (np.isin(src, ["test_harmless"]) | ((src == "xstest") & (hl == 0))))}

    return {"split": split, "pid": pid, "yn": yn, "slices": slices, "neg_mask": np.isin(pid, list(data_ids)) & (yn == 0), "train_mask": split == "train", "eval_mask": split == "eval"}


# ------------------------------------------------------------------ CPU stages


def stage_probes2(cfg, args):
    nat_rows, nat_f, syn_rows, syn_f = refusal_data(cfg, ["synthetic", "synthetic2"])
    c = refusal_context(cfg, nat_rows)
    import torch

    dirs = torch.load(cfg.path("directions.pt"))["dirs"]
    keys = ["L15_cot_first16", "L15_cot_first32", "L15_cot_mean"]
    report = sv.evaluate_task(cfg, "refusal", nat_rows, nat_f, syn_rows, syn_f, keys, dirs, c["neg_mask"], c["train_mask"], c["eval_mask"], lambda rows, ev: c["slices"](ev), c["pid"], (), "L15_eop_mean", "probes2_report")
    report["diversity"] = sv.text_diversity({g: [r for r in syn_rows if r["generator"] == g and r["label"] == 1] for g in sorted({r["generator"] for r in syn_rows})} | {"natural_refusals": [r for r in nat_rows if r["label"] == 1 and r["split"] == "train"][:1000]})
    json.dump(report, open(cfg.path("probes2_report.json"), "w"), indent=1)


def naturalness_scores(Xs, Xn_pos, pos_by_gen, groups, seed=0):
    """Cross-fitted (5 folds by prompt) discriminator between natural positives and synthetic positives; score =
    how synthetic-like each synthetic row looks (lower = more natural)."""
    from sklearn.model_selection import GroupKFold

    from natsteer.probes import make_probe

    idx = np.concatenate([p for p in pos_by_gen.values()])
    rng = np.random.default_rng(seed)
    scores = {}
    for tr, te in GroupKFold(5).split(idx, groups=groups[idx]):
        tr = rng.choice(tr, min(len(tr), 3 * len(Xn_pos)), replace=False)  # keep the classes roughly balanced
        X = np.concatenate([Xs[idx[tr]], Xn_pos])
        y = np.r_[np.ones(len(tr)), np.zeros(len(Xn_pos))].astype(int)
        s = make_probe().fit(X, y).decision_function(Xs[idx[te]])
        scores.update(zip(idx[te].tolist(), s.tolist()))
    return scores


def stage_probes2_extra(cfg, args):
    from natsteer.probe_eval import run_arm_grid

    nat_rows, nat_f, syn_rows, syn_f = refusal_data(cfg, ["synthetic", "synthetic2"])
    c = refusal_context(cfg, nat_rows)
    gen = np.array([r["generator"] for r in syn_rows])
    ys = np.array([r["label"] for r in syn_rows])
    spid = np.array([r["prompt_id"] for r in syn_rows])
    yn, pid = c["yn"], c["pid"]
    gens = [g for g in ("dom_all", "prompted", "valbundle", "valbundle2", "many_mix", "many_bundles", "dense_multi", "latent_mix", "random_bundles") if g in set(gen)]
    tr_idx, ev_idx, neg_idx = np.flatnonzero(c["train_mask"]), np.flatnonzero(c["eval_mask"]), np.flatnonzero(c["neg_mask"])
    slices = {k: v for k, v in c["slices"](ev_idx).items() if len(v) >= 20}
    report = {"generators": gens, "variants": {}}
    for pooling in ("cot_first32", "cot_mean"):
        single = f"L15_{pooling}"
        cat = [f"L{L}_{pooling}" for L in (11, 15, 19)]
        Xn1, Xs1 = nat_f[single].astype(np.float32), syn_f[single].astype(np.float32)
        ok_n, ok_s = ~np.isnan(Xn1).any(1), ~np.isnan(Xs1).any(1)
        pos_by_gen = {g: np.flatnonzero((gen == g) & (ys == 1) & ok_s) for g in gens}
        min_n = 3 if cfg.smoke else 30
        pos_by_gen = {g: p for g, p in pos_by_gen.items() if len(p) >= min_n}
        nat_pos = tr_idx[(yn[tr_idx] == 1) & ok_n[tr_idx]]
        scores = naturalness_scores(Xs1, Xn1[nat_pos], pos_by_gen, spid)
        pools = dict(pos_by_gen)
        pools["union"] = np.concatenate([p for p in pos_by_gen.values()])
        filt = {g: np.array([i for i in sorted(p.tolist(), key=lambda i: scores[i])[: max(min_n, int(0.3 * len(p)))]]) for g, p in pos_by_gen.items()}
        pools["filtered_union"] = np.concatenate(list(filt.values()))
        for g in ("dom_all", "prompted"):
            if g in filt:
                pools["filtered_" + g] = filt[g]
        for variant, names in (("single", [single]), ("multilayer", cat)):
            Xn = np.concatenate([nat_f[k].astype(np.float32) for k in names], 1)
            Xs = np.concatenate([syn_f[k].astype(np.float32) for k in names], 1)
            okn = ~np.isnan(Xn).any(1)
            tr_k, ev_k, neg_k = tr_idx[okn[tr_idx]], ev_idx[okn[ev_idx]], neg_idx[okn[neg_idx]]
            arms = make_arms(Xn, yn, tr_k, neg_k, Xs, pools, mix_names=("union", "filtered_union", "dom_all"))
            res = run_arm_grid(arms, Xn[ev_k], yn[ev_k], pid[ev_k], {k: np.isin(ev_k, v) for k, v in slices.items()}, cfg.v2_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs)
            report["variants"].setdefault(variant, {})[pooling] = res
            log(f"{variant} {pooling}: probes done ({len(arms)} arms)")
        report.setdefault("naturalness", {})[pooling] = {g: {"n": int(len(p)), "mean_score": float(np.mean([scores[i] for i in p])), "kept_mean_score": float(np.mean([scores[i] for i in filt[g]]))} for g, p in pos_by_gen.items()}
    json.dump(report, open(cfg.path("probes2_extra_report.json"), "w"), indent=1)


def stage_probes_matched(cfg, args):
    from natsteer.probe_eval import run_arm_grid

    nat_rows, nat_f = sv.load_feats(cfg, "natural", sv.load_natural(cfg))
    m_rows, m_f = sv.load_feats(cfg, "matched", "matched.jsonl")
    s_rows, s_f = sv.load_feats(cfg, "synthetic", "synthetic.jsonl")
    pid = np.array([r["prompt_id"] for r in nat_rows])
    held = np.array([wild_heldout(p) for p in pid])
    wild = np.array([r["source"] == "wildchat" for r in nat_rows])
    y = np.array([r["label"] for r in nat_rows])
    _, _, data = sv.split_prompts(cfg)
    data_ids = {p["prompt_id"] for p in data}
    mg = np.array([r["generator"] for r in m_rows])
    my = np.array([r["label"] for r in m_rows])
    sg = np.array([r["generator"] for r in s_rows])
    sy = np.array([r["label"] for r in s_rows])
    spid = np.array([r["prompt_id"] for r in s_rows])
    syn_ok = ~np.array([wild_heldout(p) for p in spid])
    ev_idx = np.flatnonzero(wild & held)
    report = {"n_eval": int(len(ev_idx)), "eval_pos": int(y[ev_idx].sum()), "matched": {g: {"n": int((mg == g).sum()), "positives": int(((mg == g) & (my == 1)).sum())} for g in sorted(set(mg))}, "features": {}}
    for key in ("L15_cot_first16", "L15_cot_first32", "L15_cot_mean"):
        Xn, Xm, Xs = nat_f[key].astype(np.float32), m_f[key].astype(np.float32), s_f[key].astype(np.float32)
        ok_n, ok_m, ok_s = ~np.isnan(Xn).any(1), ~np.isnan(Xm).any(1), ~np.isnan(Xs).any(1)
        ev = ev_idx[ok_n[ev_idx]]
        wild_tr = np.flatnonzero(wild & ~held & ok_n)
        wn_neg = wild_tr[y[wild_tr] == 0]
        # stack matched and v1 rows into one synthetic array so one arm helper serves both
        X_all = np.concatenate([Xm, Xs])
        pools = {}
        for g in sorted(set(mg)):
            pools[g] = np.flatnonzero((mg == g) & (my == 1) & ok_m)  # includes matched_base: natural refusals on the same prompts
        for g in ("dom_all", "prompted", "latent_mix"):
            if g in set(sg):
                pools[g + "_harmless_prompts"] = len(Xm) + np.flatnonzero((sg == g) & (sy == 1) & syn_ok & ok_s)
        arms = make_arms(Xn, y, wild_tr, wn_neg, X_all, {k: v for k, v in pools.items() if not k.endswith("_harmless_prompts")}, mix_names=tuple(pools))
        harmless_neg = np.flatnonzero(np.isin(pid, list(data_ids)) & ~held & (y == 0) & ok_n)
        arms |= {k: v for k, v in make_arms(Xn, y, wild_tr, harmless_neg, X_all, {k: v for k, v in pools.items() if k.endswith("_harmless_prompts")}, mix_names=tuple(pools)).items() if k != "natural"}
        report["features"][key] = run_arm_grid(arms, Xn[ev], y[ev], pid[ev], {"wildchat_heldout": np.ones(len(ev), bool)}, cfg.wild_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs, reference="natural")
        log(f"{key}: matched-prompt probes done ({len(arms)} arms)")
    json.dump(report, open(cfg.path("probes_matched_report.json"), "w"), indent=1)


def syco_data(cfg, syn_names):
    nat_rows, nat_f = sv.load_feats(cfg, "syco_natural", sv.syco_natural_rows(cfg))
    rows, feats = [], []
    for name in syn_names:
        r, f = sv.load_feats(cfg, name, name + ".jsonl")
        rows += r
        feats.append(f)
    keys = [k for k in feats[0].files if k != "ids" and all(k in f.files for f in feats)]
    return nat_rows, nat_f, rows, {k: np.concatenate([f[k] for f in feats]) for k in keys}


def stage_syco_probes2(cfg, args):
    syco_eval(cfg, ["syco_synthetic"], "syco_probes2_report.json")


def stage_syco_probes3(cfg, args):
    syco_eval(cfg, ["syco_synthetic", "syco_synthetic2"], "syco_probes3_report.json")


def syco_eval(cfg, syn_names, out_name):
    from natsteer.probe_eval import auroc_np, run_arm_grid
    from natsteer.probes import make_probe

    nat_rows, nat_f, syn_rows, syn_f = syco_data(cfg, syn_names)
    _, _, data, _ = sv.syco_state(cfg)
    data_ids = {p["prompt_id"] for p in data}
    split = np.array([r["split"] for r in nat_rows])
    pid = np.array([r["prompt_id"] for r in nat_rows])
    yn = np.array([r["label"] for r in nat_rows])
    gen = np.array([r["generator"] for r in syn_rows])
    ys = np.array([r["label"] for r in syn_rows])
    nlen = np.log1p(np.array([len(r.get("thinking") or "") for r in nat_rows]))
    slen = np.log1p(np.array([len(r.get("thinking") or "") for r in syn_rows]))
    tr_idx, ev_idx = np.flatnonzero(split == "train"), np.flatnonzero(split == "heldout")
    neg_idx = np.flatnonzero(np.isin(pid, list(data_ids)) & (yn == 0) & (split == "train"))
    gens = sorted(set(gen))
    report = {"features": {}, "length": {}}
    # CoT length: does it separate the classes, and do the generators' positives look like natural positives?
    la = make_probe().fit(nlen[tr_idx, None], yn[tr_idx]).decision_function(nlen[ev_idx, None])
    npos = nlen[tr_idx][yn[tr_idx] == 1]
    report["length"] = {"length_only_auroc_heldout": float(auroc_np(yn[ev_idx], la)), "natural_pos_median_log_chars": float(np.median(npos)), "natural_neg_median_log_chars": float(np.median(nlen[tr_idx][yn[tr_idx] == 0])), "generators": {g: {"pos_median_log_chars": float(np.median(slen[(gen == g) & (ys == 1)])) if ((gen == g) & (ys == 1)).any() else None} for g in gens}}
    edges = np.quantile(npos, np.linspace(0, 1, 11))
    nb = np.clip(np.searchsorted(edges, npos, side="right") - 1, 0, 9)
    for key in ("L15_final_mean", "L15_cot_mean", "L15_cot_first32"):
        Xn, Xs = nat_f[key].astype(np.float32), syn_f[key].astype(np.float32)
        ok_n, ok_s = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
        tr_k, ev_k, neg_k = tr_idx[ok_n[tr_idx]], ev_idx[ok_n[ev_idx]], neg_idx[ok_n[neg_idx]]
        pools = {g: np.flatnonzero((gen == g) & (ys == 1) & ok_s) for g in gens}
        rng = np.random.default_rng(0)
        for g in ("dom_all", "valbundle", "valbundle2", "latent_mix", "many_mix", "prompted"):  # synthetic positives resampled to the natural positives' CoT-length deciles
            p = pools.get(g, np.array([], int))
            if len(p) < 30:
                continue
            sb = np.clip(np.searchsorted(edges, slen[p], side="right") - 1, 0, 9)
            share = np.bincount(sb, minlength=10) / len(p)
            w = np.where(share[sb] > 0, 0.1 / np.maximum(share[sb], 1e-9), 0.0)
            pools[g + "_lenmatch"] = rng.choice(p, int(0.5 * len(p)), replace=False, p=w / w.sum())
        pools["union"] = np.concatenate([pools[g] for g in gens if g != "random_dir" and len(pools[g])])
        arms = make_arms(Xn, yn, tr_k, neg_k, Xs, pools, mix_names=("union", "valbundle", "valbundle2", "latent_mix", "many_mix", "random_dir"))
        report["features"][key] = run_arm_grid(arms, Xn[ev_k], yn[ev_k], pid[ev_k], {"all": np.ones(len(ev_k), bool)}, cfg.syco_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs)
        log(f"{key}: sycophancy probes done ({len(arms)} arms)")
    json.dump(report, open(cfg.path(out_name), "w"), indent=1)


# ------------------------------------------------------------------ sycophancy: wider latent search and more generators


def stage_syco_screen2(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("syco_attribution_wide.pt"))
    select, _, _, _ = sv.syco_state(cfg)
    v1 = json.load(open(cfg.path("syco_latents_screen_table.json")))
    conds = [{"name": "base", "specs": [], "family": "base"}]
    seen = set(v1)

    def add(L, i, lam):
        c = attr_cond(att, L, i, lam)
        if c["name"] not in seen:
            seen.add(c["name"])
            conds.append(c)

    for L in cfg.syco_new_layers:
        for i in ranked(att, L)[cfg.syco_new_ranks[0] : cfg.syco_new_ranks[1]]:
            for lam in cfg.syco_new_lams:
                add(L, i, lam)
        for i in ranked(att, L)[: cfg.syco_hi_top]:
            add(L, i, 8.0)
    for s in v1.values():
        if s.get("family") == "attr" and s["pos_rate"] >= v1["base"]["pos_rate"] + cfg.syco_near_min:
            add(s["layer"], s["latents"][0], 12.0)
    log(f"{len(conds)} sycophancy screening conditions on {len(select)} items")
    rows = sv.run_conditions(llm, conds, select, 1, cfg.syco_tokens, cfg.path("syco_screen2.jsonl"), sv.syco_labeler, keep_text=False)
    json.dump(sv.condition_table(rows), open(cfg.path("syco_screen2_table.json"), "w"), indent=1)


def stage_syco_validate2(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("syco_attribution_wide.pt"))
    _, val, _, _ = sv.syco_state(cfg)
    conds = validated_conds(cfg, att, "syco")
    log(f"{len(conds)} validation conditions on {len(val)} fresh items")
    rows = sv.run_conditions(llm, [{"name": "base", "specs": [], "family": "base"}] + conds, val, 2 if not cfg.smoke else 1, cfg.syco_tokens, cfg.path("syco_latents_validated2.jsonl"), sv.syco_labeler)
    json.dump(sv.condition_table(rows), open(cfg.path("syco_latents_validated2_table.json"), "w"), indent=1)
    _, good = good_validated(cfg, "syco")
    log(f"validated sycophancy latents: {len(good)}")


def stage_syco_generate2(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("syco_attribution_wide.pt"))
    D = torch.load(cfg.path("directions_syco.pt"))["dirs"]
    sweep = json.load(open(cfg.path("syco_sweep_table.json")))
    _, _, data, _ = sv.syco_state(cfg)
    vt, good = good_validated(cfg, "syco")
    conds = {c["name"]: c for c in validated_conds(cfg, att, "syco")}
    lat = list(good)[: cfg.max_many]
    log(f"{len(good)} validated latents, using {len(lat)}")
    gens = []
    for c, part in zip(lat, sv.partition(data, max(1, len(lat)))):
        gens.append({"name": f"many_mix:{c}", "generator": "many_mix", "specs": conds[c]["specs"], "prompts": part})
    if len(lat) >= 3:
        rng = random.Random(0)
        for b, part in enumerate(sv.partition(data, cfg.syco_n_bundles)):
            by_layer = defaultdict(lambda: 0)
            for c in rng.sample(lat, 3):
                s = vt[c]
                by_layer[s["layer"]] = by_layer[s["layer"]] + rng.uniform(0.5, 1.0) * 0.6 * sv.latent_vec(s["layer"], s["latents"][0], s["lam"], att[s["layer"]]["act_mean_active"])
            gens.append({"name": f"many_bundles:{b}", "generator": "many_bundles", "specs": [{"layer": L, "vec": v, "positions": "all"} for L, v in by_layer.items()], "prompts": part})
    base_s = sweep["base"]
    ok = sorted([(c, s) for c, s in sweep.items() if s.get("family") == "dom" and s["positions"] == "all" and sv.acceptable(s, base_s) and s["pos_rate"] >= base_s["pos_rate"] + cfg.syco_dense_min_gain], key=lambda cs: -cs[1]["pos_rate"])
    picked, seen = [], set()
    for c, s in ok:
        if (s["kind"], s["layer"]) not in seen:
            seen.add((s["kind"], s["layer"]))
            picked.append((c, s))
    picked = picked[: cfg.dense_multi_k]
    log("dense_multi settings:", [c for c, _ in picked])
    for (c, s), part in zip(picked, sv.partition(data, max(1, len(picked)))):
        gens.append({"name": f"dense_multi:{c}", "generator": "dense_multi", "specs": [{"layer": s["layer"], "vec": s["coeff"] * D[s["kind"]][s["layer"]], "positions": "all"}], "prompts": part})
    bund = sorted([c for c, s in vt.items() if s.get("family") == "bundle" and sv.acceptable(s, vt["base"]) and s["pos_rate"] >= vt["base"]["pos_rate"] + cfg.syco_gain], key=lambda c: -vt[c]["pos_rate"])
    if bund:
        log("best bundle", bund[0], vt[bund[0]]["pos_rate"])
        gens.append({"name": "valbundle2", "generator": "valbundle2", "specs": conds[bund[0]]["specs"], "prompts": data})
    json.dump({"latents": lat, "dense_multi": [c for c, _ in picked], "valbundle2": bund[:1]}, open(cfg.path("syco_generator_choices_v2.json"), "w"), indent=1)
    sv.run_conditions(llm, gens, data, 1, cfg.syco_tokens, cfg.path("syco_synthetic2.jsonl"), sv.syco_labeler)
    cfg.path("syco_synthetic2.done").write_text("ok")


def stage_syco_features2(cfg, args):
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    sv.extract_to(cfg, model, tok, read_jsonl(cfg.path("syco_synthetic2.jsonl")), "syco_synthetic2", sv.SYCO_LAYERS, sv.SYCO_POOLINGS)


# ------------------------------------------------------------------ layer sweep (extra layers, same rows)


def stage_features_layers(cfg, args):
    """Features at the extra layers for every refusal row set (suffix _L): natural, synthetic, synthetic2, matched."""
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    for name, rows in (("natural", sv.load_natural(cfg)), ("synthetic", read_jsonl(cfg.path("synthetic.jsonl"))), ("synthetic2", read_jsonl(cfg.path("synthetic2.jsonl"))), ("matched", read_jsonl(cfg.path("matched.jsonl")))):
        if not cfg.path(f"{name}_L.feats.npz").exists():
            sv.extract_to(cfg, model, tok, rows, name + "_L", cfg.layer_extra, cfg.layer_poolings)


def stage_probes_layers(cfg, args):
    """Natural vs synthetic probes per layer (7, 11, 15, 19, 23) and with all five layers concatenated."""
    from natsteer.probe_eval import run_arm_grid

    nat_rows, nat_f, syn_rows, syn_f = refusal_data(cfg, ["synthetic", "synthetic2"])
    c = refusal_context(cfg, nat_rows)
    extra_n = np.load(cfg.path("natural_L.feats.npz"))
    extra_s = [np.load(cfg.path("synthetic_L.feats.npz")), np.load(cfg.path("synthetic2_L.feats.npz"))]
    assert list(extra_n["ids"]) == [r["id"] for r in nat_rows]
    gen = np.array([r["generator"] for r in syn_rows])
    ys = np.array([r["label"] for r in syn_rows])
    yn, pid = c["yn"], c["pid"]
    gens = [g for g in ("dom_all", "prompted", "valbundle", "valbundle2", "many_mix", "many_bundles", "dense_multi") if g in set(gen)]
    tr_idx, ev_idx, neg_idx = np.flatnonzero(c["train_mask"]), np.flatnonzero(c["eval_mask"]), np.flatnonzero(c["neg_mask"])
    slices = {k: v for k, v in c["slices"](ev_idx).items() if len(v) >= 20}

    def nat(L, pooling):
        return (extra_n if L in cfg.layer_extra else nat_f)[f"L{L}_{pooling}"].astype(np.float32)

    def syn(L, pooling):
        if L in cfg.layer_extra:
            return np.concatenate([e[f"L{L}_{pooling}"] for e in extra_s]).astype(np.float32)
        return syn_f[f"L{L}_{pooling}"].astype(np.float32)

    report = {"generators": gens, "variants": {}}
    layers = (7, 11, 15, 19, 23)
    for pooling in cfg.layer_poolings:
        for variant, ls in [(f"L{L}", (L,)) for L in layers] + [("all_layers", layers)]:
            Xn = np.concatenate([nat(L, pooling) for L in ls], 1)
            Xs = np.concatenate([syn(L, pooling) for L in ls], 1)
            okn, oks = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
            tr_k, ev_k, neg_k = tr_idx[okn[tr_idx]], ev_idx[okn[ev_idx]], neg_idx[okn[neg_idx]]
            pools = {g: np.flatnonzero((gen == g) & (ys == 1) & oks) for g in gens}
            pools["union"] = np.concatenate([p for p in pools.values() if len(p)])
            arms = make_arms(Xn, yn, tr_k, neg_k, Xs, pools, mix_names=("union",))
            res = run_arm_grid(arms, Xn[ev_k], yn[ev_k], pid[ev_k], {k: np.isin(ev_k, v) for k, v in slices.items()}, cfg.v2_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs)
            report["variants"].setdefault(pooling, {})[variant] = res
            log(f"{pooling} {variant}: probes done ({len(arms)} arms)")
    json.dump(report, open(cfg.path("probes_layers_report.json"), "w"), indent=1)


def stage_summary2(cfg, args):
    import subprocess

    subprocess.run([sys.executable, str(Path(__file__).parent / "morning_summary_v2.py"), str(cfg.dir)], check=True)


def stage_report2(cfg, args):
    out = run_dir("steer_v2_smoke" if cfg.smoke else "steer_v2")
    for p in sorted(cfg.dir.glob("*.json")) + [cfg.path("SUMMARY.md")]:
        if p.exists() and not p.is_symlink():
            shutil.copy(p, out / p.name)
    (out / "report.done").write_text("ok")
    log("saved", out)


STAGES = {
    "init": (stage_init, "init.done"),
    "syco_probes2": (stage_syco_probes2, "syco_probes2_report.json"),
    "screen2": (stage_screen2, "screen2_table.json"),
    "validate2": (stage_validate2, "latents_validated2_table.json"),
    "generate2": (stage_generate2, "synthetic2.done"),
    "matched_gen": (stage_matched_gen, "matched.done"),
    "features2": (stage_features2, "matched.feats.npz"),
    "probes2": (stage_probes2, "probes2_report.json"),
    "probes2_extra": (stage_probes2_extra, "probes2_extra_report.json"),
    "probes_matched": (stage_probes_matched, "probes_matched_report.json"),
    "syco_screen2": (stage_syco_screen2, "syco_screen2_table.json"),
    "syco_validate2": (stage_syco_validate2, "syco_latents_validated2_table.json"),
    "syco_generate2": (stage_syco_generate2, "syco_synthetic2.done"),
    "syco_features2": (stage_syco_features2, "syco_synthetic2.feats.npz"),
    "syco_probes3": (stage_syco_probes3, "syco_probes3_report.json"),
    "features_layers": (stage_features_layers, "matched_L.feats.npz"),
    "probes_layers": (stage_probes_layers, "probes_layers_report.json"),
    "summary2": (stage_summary2, "summary2.never"),
    "report2": (stage_report2, "report2.never"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, outputs under data/pools/steer_v2_smoke (reads steer_v1_smoke)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = V2Cfg(args.smoke)
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
