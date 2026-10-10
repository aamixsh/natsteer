"""Refusal testbed: the steering vector defines the behavior; synthetic refusals train CoT probes.

Label = the model's own behavior (refusal phrase at the start of the final answer), no
written policy. Positives for probes come from steering harmless prompts toward refusal
(addition only; no compliance is induced on harmful prompts). Probes read activations
recomputed on the generated text without any intervention.

Stages (separate processes so vLLM and HF never share the GPU; each skips if its output exists):
  natural     vLLM, 4 samples per prompt for the 2507.03167 splits, XSTest and Aegis; labels
  directions  HF, difference-of-means refusal directions at every layer (end-of-prompt, CoT,
              and the dataset-defined harmful-vs-harmless end-of-prompt direction)
  sweep       vLLM hooks, addition on harmless selection prompts: kind x layer x positions x coeff,
              matched-norm random controls
  sweep_hf    HF Steerer, CoT-only and final-only conditions for the best directions
  attribution HF, Delta-attribution of SAE latents (layers 15, 19) on natural refuse/comply pairs
  latents     vLLM hooks, single top latents, Delta-activation and random latents, weighted bundles
  generate    vLLM, synthetic data: best directions, best bundle, a diverse latent mix, random and
              prompted controls, on harmless training prompts
  features    HF, clean pooled features of natural and synthetic samples
  probes      CPU, probes on natural vs synthetic vs mixed data, evaluated on natural held-out samples
  syco_natural, syco_attr   sycophancy add-on: mixed pairs and Delta-attribution only
Usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/refusal_pipeline.py --stage natural [--smoke]
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from natsteer.config import DATA_ROOT, run_dir
from natsteer.harmony import render
from natsteer.refusal import (OUT, compression_ratio, distinct_n, label_generation, load_prompts, read_jsonl, write_jsonl)

EFFORT = "medium"
DIR_LAYERS = tuple(range(24))
SWEEP_LAYERS = (11, 15, 19)
SAE_LAYERS = (15, 19)
FEAT_LAYERS = (7, 11, 15, 19, 23)
FEAT_POOLINGS = ("prompt_last", "eop_mean", "cot_mean", "cot_first32", "final_mean")
PROMPTED_SUFFIX = "\n\n(Politely decline to help with this request.)"


class Cfg:
    def __init__(self, smoke: bool):
        self.smoke = smoke
        self.out = OUT.parent / "refusal_smoke" if smoke else OUT
        self.n_samples = 2 if smoke else 4
        self.max_tokens = 1024 if smoke else 1536
        self.steer_max_tokens = 384 if smoke else 1024
        self.n_select = 8 if smoke else 96
        self.select_samples = 1 if smoke else 2
        # Multiples of the raw difference-of-means vector. Smoke run: all-position steering broke
        # generation at 4 while prompt-only steering at 4 was still coherent, so the grids differ.
        self.coeffs = {"all": (1.0, 4.0), "prompt": (1.0, 4.0)} if smoke else {"all": (0.5, 1.0, 1.5, 2.0, 3.0), "prompt": (1.0, 2.0, 4.0, 6.0, 8.0)}
        self.sweep_layers = (15,) if smoke else SWEEP_LAYERS
        self.max_pairs = 6 if smoke else 400
        self.n_top_latents = 2 if smoke else 12
        self.latent_lams = (5.0,) if smoke else (1.0, 2.0, 5.0)  # smoke: single latents at 5x their mean active value often broke generation
        self.n_data_prompts = 16 if smoke else 100000
        self.hf_prompts = 4 if smoke else 32
        self.budgets = (2, 4) if smoke else (64, 128, 256, 512, 1024)

    def path(self, name):
        self.out.mkdir(parents=True, exist_ok=True)
        return self.out / name


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def prompts_for(cfg):
    rows = load_prompts()
    if cfg.smoke:
        by = defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        rows = [r for src in by for r in by[src][:12]]
    return rows


# ---------------------------------------------------------------- natural


def stage_natural(cfg, args):
    from vllm import SamplingParams

    from natsteer.config import MODEL_ID
    from natsteer.vllm_utils import make_llm

    rows = prompts_for(cfg)
    llm = make_llm(MODEL_ID, max_model_len=4096, gpu_memory_utilization=0.88)
    tok = llm.get_tokenizer()
    params = SamplingParams(n=cfg.n_samples, temperature=1.0, max_tokens=cfg.max_tokens, seed=0)
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": render(tok, r["user"], effort=EFFORT)[0]} for r in rows], params)
    samples = []
    for r, o in zip(rows, outs):
        for k, c in enumerate(o.outputs):
            samples.append(r | {"id": f"{r['prompt_id']}#s{k}", "effort": EFFORT, "finish": c.finish_reason} | label_generation(tok, c.token_ids))
    write_jsonl(cfg.path("natural.jsonl"), samples)
    ntok = sum(s["n_gen_tokens"] for s in samples)
    log(f"natural: {len(samples)} samples, {ntok} tokens in {time.time() - t0:.0f}s ({ntok / (time.time() - t0):.0f} tok/s)")
    for src in sorted({s["source"] for s in samples}):
        lab = [s["refusal"] for s in samples if s["source"] == src]
        log(f"  {src}: refusal {lab.count(1)}, comply {lab.count(0)}, incomplete {lab.count(None)}")


def prompt_table(samples):
    """prompt_id -> {source, user, n_ref, n_comp, n_inc}"""
    t = {}
    for s in samples:
        d = t.setdefault(s["prompt_id"], {"source": s["source"], "user": s["user"], "split": s["split"], "n_ref": 0, "n_comp": 0, "n_inc": 0})
        d["n_ref" if s["refusal"] == 1 else "n_comp" if s["refusal"] == 0 else "n_inc"] += 1
    return t


def selection_and_data_prompts(cfg, samples):
    """Harmless training prompts the model always answers: a selection set for the sweeps, the rest for data."""
    t = prompt_table(samples)
    ok = sorted(pid for pid, d in t.items() if d["source"] == "train_harmless" and d["n_ref"] == 0 and d["n_comp"] >= 2)
    random.Random(0).shuffle(ok)
    select = ok[: cfg.n_select]
    data = ok[cfg.n_select :][: cfg.n_data_prompts]
    return [{"prompt_id": p, "user": t[p]["user"]} for p in select], [{"prompt_id": p, "user": t[p]["user"]} for p in data]


# ---------------------------------------------------------------- directions


def stage_directions(cfg, args):
    import torch

    from natsteer.features import extract
    from natsteer.model import load_model

    samples = read_jsonl(cfg.path("natural.jsonl"))
    t = prompt_table(s for s in samples if s["split"] == "train")
    model, tok = load_model("bf16")
    res, info = {}, {}

    def frac(d):
        n = d["n_ref"] + d["n_comp"]
        return d["n_ref"] / n if n >= 2 else None

    # Prompt level, behavior-defined: prompts refused in >= 75% of samples vs <= 25%.
    ref = [{"id": p, "user": d["user"]} for p, d in t.items() if frac(d) is not None and frac(d) >= 0.75]
    comp = [{"id": p, "user": d["user"]} for p, d in t.items() if frac(d) is not None and frac(d) <= 0.25]
    # Prompt level, dataset-defined (Arditi et al.): harmful vs harmless training prompts.
    harm = [{"id": p, "user": d["user"]} for p, d in t.items() if d["source"] == "train_harmful"]
    harmless = [{"id": p, "user": d["user"]} for p, d in t.items() if d["source"] == "train_harmless"]
    feats = {}
    for name, rows in (("ref", ref), ("comp", comp), ("harm", harm), ("harmless", harmless)):
        f = extract(model, tok, rows, DIR_LAYERS, ("eop_mean",), log_every=0)
        feats[name] = np.stack([f[L]["eop_mean"].astype(np.float32).mean(0) for L in DIR_LAYERS])
    res["eop"] = torch.tensor(feats["ref"] - feats["comp"])
    res["eop_dataset"] = torch.tensor(feats["harm"] - feats["harmless"])
    info["eop"] = {"n_ref": len(ref), "n_comp": len(comp)}
    info["eop_dataset"] = {"n_harm": len(harm), "n_harmless": len(harmless)}
    # CoT level: mean over CoT tokens of refusal samples vs compliance samples.
    rng = random.Random(0)
    rs = [s for s in samples if s["split"] == "train" and s["refusal"] == 1 and s["thinking"]]
    cs = [s for s in samples if s["split"] == "train" and s["refusal"] == 0 and s["thinking"]]
    rs, cs = rng.sample(rs, min(len(rs), 800)), rng.sample(cs, min(len(cs), 800))
    cot = {}
    for name, rows in (("ref", rs), ("comp", cs)):
        rows = [{"id": s["id"], "user": s["user"], "thinking": s["thinking"], "effort": EFFORT} for s in rows]
        f = extract(model, tok, rows, DIR_LAYERS, ("cot_mean",), max_len=2048, log_every=0)
        cot[name] = np.stack([np.nanmean(f[L]["cot_mean"].astype(np.float32), 0) for L in DIR_LAYERS])
    res["cot"] = torch.tensor(cot["ref"] - cot["comp"])
    info["cot"] = {"n_ref": len(rs), "n_comp": len(cs)}
    # Scale references: typical residual norm at the end of the prompt per layer.
    info["resid_norm_eop"] = [float(np.linalg.norm(feats["comp"][L])) for L in DIR_LAYERS]
    info["dir_norm"] = {k: [float(v[L].norm()) for L in DIR_LAYERS] for k, v in res.items()}
    unit = {k: v / v.norm(dim=1, keepdim=True) for k, v in res.items()}
    info["cos"] = {f"{a}~{b}": [float((unit[a][L] * unit[b][L]).sum()) for L in DIR_LAYERS] for a, b in (("eop", "cot"), ("eop", "eop_dataset"), ("cot", "eop_dataset"))}
    torch.save({"dirs": res, "info": info}, cfg.path("directions.pt"))
    log("directions:", json.dumps({k: v for k, v in info.items() if k in ("eop", "cot", "eop_dataset")}))
    for L in SWEEP_LAYERS:
        log(f"  L{L}: norms eop {info['dir_norm']['eop'][L]:.1f} cot {info['dir_norm']['cot'][L]:.1f} resid {info['resid_norm_eop'][L]:.1f}; cos eop~cot {info['cos']['eop~cot'][L]:.2f} eop~dataset {info['cos']['eop~eop_dataset'][L]:.2f}")


# ---------------------------------------------------------------- steered generation helpers


def summarize(rows):
    lab = [r["refusal"] for r in rows]
    n = len(rows)
    if n == 0:
        return {"n": 0, "refusal_rate": 0.0, "incomplete_rate": 1.0, "cot_refusal_rate": 0.0, "rep4": 1.0, "gen_tokens": 0.0, "final_distinct2": 0.0, "final_compression": 0.0}
    finals = [r["final"] for r in rows if r["final"]]
    return {
        "n": n,
        "refusal_rate": lab.count(1) / n,
        "incomplete_rate": lab.count(None) / n,
        "cot_refusal_rate": float(np.mean([r["cot_refusal"] for r in rows])),
        "rep4": float(np.mean([r["rep4"] for r in rows])),
        "gen_tokens": float(np.mean([r["n_gen_tokens"] for r in rows])),
        "final_distinct2": distinct_n(finals, 2),
        "final_compression": compression_ratio(finals),
    }


def run_conditions(cfg, llm, conditions, prompts, n, max_tokens, out_name):
    """conditions: list of dicts {name, specs, ...meta}; appends labeled rows per condition (resumable)."""
    from vllm import SamplingParams

    from natsteer.vllm_steer import set_steering

    tok = llm.get_tokenizer()
    path = cfg.path(out_name)
    done = {r["condition"] for r in read_jsonl(path)}
    rows_all = read_jsonl(path)
    reqs = [{"prompt_token_ids": render(tok, p.get("gen_user", p["user"]), effort=EFFORT)[0]} for p in prompts]
    for c in conditions:
        if c["name"] in done:
            continue
        set_steering(llm, c["specs"])
        t0 = time.time()
        outs = llm.generate(reqs, SamplingParams(n=n, temperature=1.0, max_tokens=max_tokens, seed=1), use_tqdm=False)
        meta = {k: v for k, v in c.items() if k != "specs"}
        rows = []
        for p, o in zip(prompts, outs):
            for k, comp in enumerate(o.outputs):
                rows.append({"id": f"{p['prompt_id']}|{c['name']}#s{k}", "prompt_id": p["prompt_id"], "user": p["user"], "effort": EFFORT, "condition": c["name"], **meta} | label_generation(tok, comp.token_ids))
        rows_all += rows
        done.add(c["name"])
        write_jsonl(path, rows_all)
        s = summarize(rows)
        log(f"{c['name']}: refusal {s['refusal_rate']:.2f} incomplete {s['incomplete_rate']:.2f} rep4 {s['rep4']:.3f} tokens {s['gen_tokens']:.0f} ({time.time() - t0:.0f}s)")
    set_steering(llm, [])
    return rows_all


def condition_table(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    return {c: summarize(rs) | {k: rs[0].get(k) for k in ("kind", "layer", "positions", "coeff", "latents", "lam", "family")} for c, rs in by.items()}


def acceptable(s, base, inc_slack=0.15, rep_slack=0.05):
    return s["incomplete_rate"] <= base["incomplete_rate"] + inc_slack and s["rep4"] <= base["rep4"] + rep_slack


def best(table, family_filter, base, min_rate=0.0):
    cands = [(c, s) for c, s in table.items() if family_filter(c, s) and acceptable(s, base) and s["refusal_rate"] >= min_rate]
    return max(cands, key=lambda cs: cs[1]["refusal_rate"]) if cands else None


# ---------------------------------------------------------------- sweep


def stage_sweep(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    samples = read_jsonl(cfg.path("natural.jsonl"))
    select, _ = selection_and_data_prompts(cfg, samples)
    write_jsonl(cfg.path("selection_prompts.jsonl"), select)
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    g = torch.randn(2880, generator=torch.Generator().manual_seed(0))
    rand = g / g.norm()
    conds = [{"name": "base", "specs": [], "family": "base"}]
    for kind in ("eop", "cot"):
        for L in cfg.sweep_layers:
            for pos in ("all", "prompt"):
                for c in cfg.coeffs[pos]:
                    conds.append({"name": f"{kind}_L{L}_{pos}_c{c}", "kind": kind, "layer": L, "positions": pos, "coeff": c, "family": "dom", "specs": [{"layer": L, "vec": c * D[kind][L], "positions": pos}]})
    for L in cfg.sweep_layers:
        for pos in ("all", "prompt"):
            for c in sorted({cfg.coeffs[pos][1], cfg.coeffs[pos][-1]}):
                conds.append({"name": f"random_L{L}_{pos}_c{c}", "kind": "random", "layer": L, "positions": pos, "coeff": c, "family": "random", "specs": [{"layer": L, "vec": c * float(D["eop"][L].norm()) * rand, "positions": pos}]})
    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    rows = run_conditions(cfg, llm, conds, select, cfg.select_samples, cfg.steer_max_tokens, "sweep.jsonl")
    table = condition_table(rows)
    json.dump(table, open(cfg.path("sweep_table.json"), "w"), indent=1)


# ---------------------------------------------------------------- sweep_hf (CoT-only, final-only)


def stage_sweep_hf(cfg, args):
    import torch

    from natsteer.model import Steerer, generate, load_model

    table = json.load(open(cfg.path("sweep_table.json")))
    base = table["base"]
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    select = read_jsonl(cfg.path("selection_prompts.jsonl"))[: cfg.hf_prompts]
    picks = []
    for kind in ("eop", "cot"):
        b = best(table, lambda c, s, kind=kind: s.get("kind") == kind and s.get("positions") == "all", base)
        if b:
            picks.append(b[1])
    model, tok = load_model("bf16")
    prompts = [render(tok, p["user"], effort=EFFORT)[0] for p in select]
    path = cfg.path("sweep_hf.jsonl")
    rows_all = read_jsonl(path)
    done = {r["condition"] for r in rows_all}
    for s in picks:
        for coeff in (s["coeff"], 2 * s["coeff"]):
            for segs in (("cot",), ("final",), None):
                name = f"hf_{s['kind']}_L{s['layer']}_{'+'.join(segs) if segs else 'all'}_c{coeff}"
                if name in done:
                    continue
                vec = (coeff * D[s["kind"]][s["layer"]]).to(model.device, torch.bfloat16)
                t0 = time.time()
                with Steerer(model, tok, s["layer"], vec, segments=segs) as st:
                    gens = generate(model, tok, prompts, max_new_tokens=cfg.steer_max_tokens, temperature=1.0, steerer=st, batch_size=32, seed=0)
                rows = [{"id": f"{p['prompt_id']}|{name}", "prompt_id": p["prompt_id"], "user": p["user"], "condition": name, "kind": s["kind"], "layer": s["layer"], "positions": "+".join(segs) if segs else "all_hf", "coeff": coeff, "family": "dom_hf"} | label_generation(tok, g) for p, g in zip(select, gens)]
                rows_all += rows
                write_jsonl(path, rows_all)
                sm = summarize(rows)
                log(f"{name}: refusal {sm['refusal_rate']:.2f} incomplete {sm['incomplete_rate']:.2f} rep4 {sm['rep4']:.3f} ({time.time() - t0:.0f}s)")
    json.dump(condition_table(rows_all), open(cfg.path("sweep_hf_table.json"), "w"), indent=1)


# ---------------------------------------------------------------- attribution


def pairs_from(samples, key, max_pairs, seed=0):
    """Same-prompt pairs (behavior sample, contrast sample) where `key` is 1 vs 0."""
    by = defaultdict(lambda: ([], []))
    for s in samples:
        if s.get(key) in (0, 1):
            by[s["prompt_id"]][1 - s[key]].append(s)
    pairs = []
    for pid, (pos, neg) in sorted(by.items()):
        for a, b in zip(pos, neg):
            pairs.append((a, b))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]


def run_attribution(cfg, pairs, out_name):
    import torch

    from natsteer.attribution import delta_attribution
    from natsteer.model import load_model
    from natsteer.sae import load_andyrdt_sae

    model, tok = load_model("bf16")
    saes = {L: load_andyrdt_sae(L, 64, device=model.device) for L in SAE_LAYERS}
    log(f"attribution on {len(pairs)} pairs")
    res = delta_attribution(model, tok, pairs, saes, log_every=25)
    D = None
    try:
        D = torch.load(cfg.path("directions.pt"))["dirs"]
    except FileNotFoundError:
        pass
    summary = {"n_pairs": res["n_pairs"]}
    for L in SAE_LAYERS:
        r = res[L]
        top = torch.argsort(r["delta_attr"], descending=True)[:64]
        top_act = torch.argsort(r["delta_act"], descending=True)[:64]
        Wd = saes[L].W_dec.float().cpu()
        entry = {
            "top_delta_attr": [{"latent": int(i), "score": float(r["delta_attr"][i]), "t": float(r["delta_attr_t"][i]), "delta_act": float(r["delta_act"][i]), "act_mean_active": float(r["act_mean_active"][i]), "freq": float(r["act_freq"][i])} for i in top],
            "top_delta_act": [{"latent": int(i), "delta_act": float(r["delta_act"][i]), "score": float(r["delta_attr"][i])} for i in top_act],
            "overlap_top20": len(set(top[:20].tolist()) & set(top_act[:20].tolist())),
        }
        if D is not None:
            for kind in ("eop", "cot"):
                d = D[kind][L].float()
                Wn = Wd / Wd.norm(dim=1, keepdim=True)
                cos = Wn @ (d / d.norm())
                entry[f"cos_{kind}_top_attr"] = [float(cos[i]) for i in top[:20]]
                entry[f"top_cos_{kind}"] = [{"latent": int(i), "cos": float(cos[i])} for i in torch.argsort(cos, descending=True)[:20]]
                # How much of the difference-of-means vector the top-k attributed decoder rows span.
                for k in (8, 32, 64):
                    B = Wd[top[:k]].T
                    coef = torch.linalg.lstsq(B, d).solution
                    entry[f"r2_{kind}_top{k}"] = float(1 - ((B @ coef - d).norm() / d.norm()) ** 2)
        summary[f"L{L}"] = entry
        log(f"L{L}: top attr latents {[e['latent'] for e in entry['top_delta_attr'][:10]]}; overlap with delta-act top20 {entry['overlap_top20']}")
    torch.save({L: {k: v for k, v in res[L].items()} for L in SAE_LAYERS} | {"n_pairs": res["n_pairs"]}, cfg.path(out_name + ".pt"))
    json.dump(summary, open(cfg.path(out_name + ".json"), "w"), indent=1)


def stage_attribution(cfg, args):
    samples = [s for s in read_jsonl(cfg.path("natural.jsonl")) if s["split"] == "train" and s["final"]]
    pairs = pairs_from(samples, "refusal", cfg.max_pairs)
    run_attribution(cfg, pairs, "attribution")


# ---------------------------------------------------------------- latents


def latent_vec(saes_dec, L, latent, scale):
    return scale * saes_dec[L][latent]


def stage_latents(cfg, args):
    import torch

    from natsteer.sae import load_andyrdt_sae
    from natsteer.vllm_steer import make_steerable_llm

    # Create the engine first: vLLM forks its engine process, and a fork taken after large CPU torch
    # ops (SAE construction) has hung at start-up here (2026-10-10, twice, in this stage).
    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("attribution.pt"))
    summary = json.load(open(cfg.path("attribution.json")))
    select = read_jsonl(cfg.path("selection_prompts.jsonl"))
    dec = {L: load_andyrdt_sae(L, 64, device="cpu").W_dec.float() for L in SAE_LAYERS}
    rng = np.random.default_rng(0)
    conds = [{"name": "base", "specs": [], "family": "base"}]
    for L in SAE_LAYERS:
        r = att[L]
        top = [e["latent"] for e in summary[f"L{L}"]["top_delta_attr"]][: cfg.n_top_latents]
        top_act = [e["latent"] for e in summary[f"L{L}"]["top_delta_act"] if e["latent"] not in top][:4 if not cfg.smoke else 1]
        alive = np.flatnonzero(r["act_freq"].numpy() > 1e-4)
        rand_lat = [int(i) for i in rng.choice(alive, 4 if not cfg.smoke else 1, replace=False)]
        scale = r["act_mean_active"]
        for fam, lats, lams in (("attr", top, cfg.latent_lams), ("act", top_act, cfg.latent_lams[1:] or cfg.latent_lams), ("randlat", rand_lat, cfg.latent_lams[1:] or cfg.latent_lams)):
            for i in lats:
                for lam in lams:
                    v = latent_vec(dec, L, i, lam * float(scale[i]))
                    conds.append({"name": f"{fam}_L{L}_{i}_x{lam}", "family": fam, "layer": L, "latents": [i], "lam": lam, "positions": "all", "specs": [{"layer": L, "vec": v, "positions": "all"}]})
        for k in ((4, 8) if not cfg.smoke else (2,)):
            w = torch.tensor([max(0.0, float(r["delta_attr"][i])) for i in top[:k]])
            w = w / w.max()
            for lam in cfg.latent_lams:
                v = sum(lam * float(w[j]) * float(scale[i]) * dec[L][i] for j, i in enumerate(top[:k]))
                for pos in ("all", "prompt"):
                    conds.append({"name": f"bundle{k}_L{L}_{pos}_x{lam}", "family": "bundle", "layer": L, "latents": top[:k], "lam": lam, "positions": pos, "specs": [{"layer": L, "vec": v, "positions": pos}]})
    rows = run_conditions(cfg, llm, conds, select, 1, cfg.steer_max_tokens, "latents.jsonl")
    json.dump(condition_table(rows), open(cfg.path("latents_table.json"), "w"), indent=1)


# ---------------------------------------------------------------- generate (synthetic data)


def stage_generate(cfg, args):
    import torch

    from natsteer.sae import load_andyrdt_sae
    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)  # before any CPU torch work, see stage_latents
    samples = read_jsonl(cfg.path("natural.jsonl"))
    _, data = selection_and_data_prompts(cfg, samples)
    write_jsonl(cfg.path("data_prompts.jsonl"), data)
    sweep = json.load(open(cfg.path("sweep_table.json")))
    lat = json.load(open(cfg.path("latents_table.json")))
    base_s, base_l = sweep["base"], lat["base"]
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    att = torch.load(cfg.path("attribution.pt"))
    dec = {L: load_andyrdt_sae(L, 64, device="cpu").W_dec.float() for L in SAE_LAYERS}
    choices = {}
    plan = []  # (generator name, condition dict, prompts)

    def dom_spec(s):
        return [{"layer": s["layer"], "vec": s["coeff"] * D[s["kind"]][s["layer"]], "positions": s["positions"]}]

    b = best(sweep, lambda c, s: s.get("family") == "dom" and s.get("positions") == "all", base_s, 0.2)
    if b:
        choices["dom_all"] = b[0]
        plan.append(("dom_all", {"name": "dom_all", "specs": dom_spec(b[1])}, data))
        g = torch.randn(2880, generator=torch.Generator().manual_seed(0))
        v = g / g.norm() * float((b[1]["coeff"] * D[b[1]["kind"]][b[1]["layer"]]).norm())
        plan.append(("random_dir", {"name": "random_dir", "specs": [{"layer": b[1]["layer"], "vec": v, "positions": "all"}]}, data))
    b = best(sweep, lambda c, s: s.get("family") == "dom" and s.get("positions") == "prompt", base_s, 0.2)
    if b:
        choices["dom_prompt"] = b[0]
        plan.append(("dom_prompt", {"name": "dom_prompt", "specs": dom_spec(b[1])}, data))
    b = best(lat, lambda c, s: s.get("family") == "bundle", base_l, 0.2)
    if b:
        s = b[1]
        L, lats = s["layer"], s["latents"]
        r = att[L]
        w = torch.tensor([max(0.0, float(r["delta_attr"][i])) for i in lats])
        w = w / w.max()
        v = sum(s["lam"] * float(w[j]) * float(r["act_mean_active"][i]) * dec[L][i] for j, i in enumerate(lats))
        choices["bundle"] = b[0]
        plan.append(("bundle", {"name": "bundle", "specs": [{"layer": L, "vec": v, "positions": s["positions"]}]}, data))
    # Diverse mix: every validated single latent (best lambda each), prompts split among them.
    valid = {}
    for c, s in lat.items():
        if s.get("family") == "attr" and acceptable(s, base_l) and s["refusal_rate"] >= max(0.2, base_l["refusal_rate"] + 0.1):
            key = (s["layer"], s["latents"][0])
            if key not in valid or s["refusal_rate"] > valid[key][1]["refusal_rate"]:
                valid[key] = (c, s)
    choices["diverse"] = sorted(c for c, _ in valid.values())
    if valid:
        order = list(valid.values())
        parts = [data[i :: len(order)] for i in range(len(order))]
        for (c, s), part in zip(order, parts):
            L, i = s["layer"], s["latents"][0]
            v = s["lam"] * float(att[L]["act_mean_active"][i]) * dec[L][i]
            plan.append(("diverse", {"name": f"diverse:{c}", "specs": [{"layer": L, "vec": v, "positions": "all"}]}, part))
    prompted = [p | {"gen_user": p["user"] + PROMPTED_SUFFIX} for p in data]
    plan.append(("prompted", {"name": "prompted", "specs": []}, prompted))
    json.dump(choices, open(cfg.path("generator_choices.json"), "w"), indent=1)
    log("generators:", json.dumps(choices))
    for gen, cond, prompts in plan:
        run_conditions(cfg, llm, [cond | {"generator": gen}], prompts, 1, cfg.max_tokens, "synthetic.jsonl")
    cfg.path("synthetic.done").write_text("ok")


# ---------------------------------------------------------------- features


def stage_features(cfg, args):
    from natsteer.features import extract, save
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    for name in ("natural", "synthetic"):
        if cfg.path(f"{name}.feats.npz").exists():
            continue
        rows = [r for r in read_jsonl(cfg.path(f"{name}.jsonl")) if r["refusal"] is not None]
        t0 = time.time()
        save(cfg.path(f"{name}.feats.npz"), rows, extract(model, tok, rows, FEAT_LAYERS, FEAT_POOLINGS, max_len=3072, log_every=2000))
        log(f"features {name}: {len(rows)} rows in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- probes


def stage_probes(cfg, args):
    import torch

    from natsteer.probes import auroc, make_probe, oof_scores, tpr_at_fpr

    nat = [r for r in read_jsonl(cfg.path("natural.jsonl")) if r["refusal"] is not None]
    syn = [r for r in read_jsonl(cfg.path("synthetic.jsonl")) if r["refusal"] is not None]
    fn, fs = np.load(cfg.path("natural.feats.npz")), np.load(cfg.path("synthetic.feats.npz"))
    assert list(fn["ids"]) == [r["id"] for r in nat] and list(fs["ids"]) == [r["id"] for r in syn]
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    data_ids = {p["prompt_id"] for p in read_jsonl(cfg.path("data_prompts.jsonl"))}
    report = {"n_natural": len(nat), "n_synthetic": len(syn)}

    ytrain_idx = np.array([i for i, r in enumerate(nat) if r["split"] == "train"])
    eval_idx = np.array([i for i, r in enumerate(nat) if r["split"] == "eval"])
    yn = np.array([r["refusal"] for r in nat])
    src = np.array([r["source"] for r in nat])
    # Natural negatives for synthetic arms: unsteered compliant samples on the same data prompts.
    nat_neg_data = np.array([i for i, r in enumerate(nat) if r["prompt_id"] in data_ids and r["refusal"] == 0])
    gens = sorted({r["generator"] for r in syn})
    gen_arr = np.array([r["generator"] for r in syn])
    ys = np.array([r["refusal"] for r in syn])
    report["synthetic_yield"] = {g: {"n": int((gen_arr == g).sum()), "refusal_rate": float(ys[gen_arr == g].mean())} for g in gens}
    budgets = list(cfg.budgets)
    seeds = 2 if cfg.smoke else 3
    slices = {
        "all": eval_idx,
        "harmful_prompts": eval_idx[np.isin(src[eval_idx], ["test_harmful"]) | ((src[eval_idx] == "xstest") & np.array([nat[i]["human_label"] == 1 for i in eval_idx]))],
        "harmless_prompts": eval_idx[np.isin(src[eval_idx], ["test_harmless"]) | ((src[eval_idx] == "xstest") & np.array([nat[i]["human_label"] == 0 for i in eval_idx]))],
        "xstest": eval_idx[src[eval_idx] == "xstest"],
    }
    report["eval_slices"] = {k: {"n": int(len(v)), "refusal_rate": float(yn[v].mean()) if len(v) else None} for k, v in slices.items()}

    def metrics(scores, idx):
        y = yn[idx]
        if len(idx) == 0 or len(set(y)) < 2:
            return None
        s = scores[np.searchsorted(eval_idx, idx)]
        return {"auroc": auroc(y, s), "tpr@0.01": tpr_at_fpr(y, s, 0.01), "tpr@0.05": tpr_at_fpr(y, s, 0.05)}

    keys = [k for k in ("L15_cot_mean", "L15_cot_first32", "L19_cot_mean", "L15_final_mean") if k in fn.files]
    report["features"] = {}
    for key in keys:
        Xn, Xs = fn[key].astype(np.float32), fs[key].astype(np.float32)
        okn, oks = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
        tr = ytrain_idx[okn[ytrain_idx]]
        ev = eval_idx[okn[eval_idx]]
        Xev = Xn[eval_idx]
        Xev = np.where(np.isnan(Xev), 0, Xev)
        L = int(key.split("_")[0][1:])
        proj = {}
        for kind in ("eop", "cot"):
            u = D[kind][L].float().numpy()
            proj[kind] = u / np.linalg.norm(u)

        def fit_eval(X, y, project=None):
            if len(set(y)) < 2:
                return None
            Xe = Xev
            if project is not None:
                X = X - np.outer(X @ project, project)
                Xe = Xe - np.outer(Xe @ project, project)
            s = make_probe().fit(X, y).decision_function(Xe)
            return {sl: metrics(s, idx[np.isin(idx, ev)]) for sl, idx in slices.items()}

        arms = defaultdict(lambda: defaultdict(list))
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            nat_perm = rng.permutation(tr)
            negs = rng.permutation(nat_neg_data)
            negs = negs[okn[negs]]
            for b in budgets:
                arms["natural"][b].append(fit_eval(Xn[nat_perm[:b]], yn[nat_perm[:b]]))
                for g in gens:
                    pos = rng.permutation(np.flatnonzero((gen_arr == g) & (ys == 1) & oks))[: b // 2]
                    if len(pos) < b // 2 or len(negs) < b // 2:
                        continue
                    X = np.concatenate([Xs[pos], Xn[negs[: b // 2]]])
                    y = np.r_[np.ones(len(pos)), np.zeros(b // 2)].astype(int)
                    arms[f"synthetic_{g}"][b].append(fit_eval(X, y))
                    half = nat_perm[: b // 2]
                    Xm = np.concatenate([Xn[half], Xs[pos[: b // 4]], Xn[negs[: b // 4]]])
                    ym = np.r_[yn[half], np.ones(b // 4), np.zeros(b // 4)].astype(int)
                    arms[f"mix_{g}"][b].append(fit_eval(Xm, ym))
                    if g == "dom_all" and seed == 0:
                        for kind in ("eop", "cot"):
                            arms[f"synthetic_{g}_minus_{kind}dir"][b].append(fit_eval(X, y, proj[kind]))

        def agg(results):
            out = {}
            for sl in slices:
                vals = [r[sl] for r in results if r and r.get(sl)]
                if vals:
                    out[sl] = {m: float(np.mean([v[m] for v in vals])) for m in ("auroc", "tpr@0.01", "tpr@0.05")}
            return out

        feat = {"arms": {a: {b: agg(rs) for b, rs in d.items()} for a, d in arms.items()}}
        feat["natural_all_train"] = fit_eval(Xn[tr], yn[tr])
        # Naturalness: discriminator AUC between synthetic and natural refusals (clean features).
        # Naturalness: discriminator AUC between synthetic refusals and natural refusals (clean
        # features). Natural refusals are mostly on harmful prompts, so "all" is confounded by topic;
        # "harmless" compares with natural over-refusals on harmless training prompts only.
        harmless_lab = np.array([r["human_label"] == 0 for r in nat])
        refs = {"all": tr[yn[tr] == 1], "harmless": tr[(yn[tr] == 1) & harmless_lab[tr]]}
        disc = {"n_natural_refusals": {k: int(len(v)) for k, v in refs.items()}}
        for g in gens:
            pos = np.flatnonzero((gen_arr == g) & (ys == 1) & oks)
            for rk, nat_ref in refs.items():
                if len(pos) < 10 or len(nat_ref) < 10:
                    continue
                m = min(len(pos), len(nat_ref), 500)
                rng = np.random.default_rng(0)
                a, bb = Xs[rng.choice(pos, m, replace=False)], Xn[rng.choice(nat_ref, m, replace=False)]
                X, y = np.concatenate([a, bb]), np.r_[np.ones(m), np.zeros(m)].astype(int)
                disc[f"{g}_vs_{rk}"] = auroc(y, oof_scores(X, y, folds=min(5, m)))
        feat["discriminator_auc"] = disc
        report["features"][key] = feat
        log(f"{key}: done")

    # Text diversity of verified synthetic refusals vs natural refusals.
    div = {}
    for g in gens:
        rows = [r for r in syn if r["generator"] == g and r["refusal"] == 1][:500]
        div[g] = {"n": len(rows), "final_compression": compression_ratio(r["final"] for r in rows), "final_distinct2": distinct_n([r["final"] for r in rows], 2), "cot_compression": compression_ratio(r["thinking"] for r in rows), "cot_distinct2": distinct_n([r["thinking"] for r in rows], 2)}
    rows = [nat[i] for i in ytrain_idx if yn[i] == 1][:500]
    div["natural_refusals"] = {"n": len(rows), "final_compression": compression_ratio(r["final"] for r in rows), "final_distinct2": distinct_n([r["final"] for r in rows], 2), "cot_compression": compression_ratio(r["thinking"] for r in rows), "cot_distinct2": distinct_n([r["thinking"] for r in rows], 2)}
    report["diversity"] = div
    json.dump(report, open(cfg.path("probes_report.json"), "w"), indent=1)


def stage_report(cfg, args):
    out = run_dir("refusal_pipeline_smoke" if cfg.smoke else "refusal_pipeline")
    for f in ("probes_report.json", "sweep_table.json", "sweep_hf_table.json", "latents_table.json", "attribution.json", "generator_choices.json", "syco_attribution.json", "timings.log"):
        if cfg.path(f).exists():
            (out / f).write_text(cfg.path(f).read_text())
    (out / "report.done").write_text("ok")
    log("saved", out)


# ---------------------------------------------------------------- sycophancy add-on


def stage_syco_natural(cfg, args):
    from vllm import SamplingParams

    from natsteer.config import MODEL_ID
    from natsteer.vllm_utils import make_llm

    sys.path.insert(0, str(Path(__file__).parent))
    from syco_pipeline import label_generation as syco_label

    items = read_jsonl(DATA_ROOT / "pools" / "syco_v0" / "train.jsonl")
    items = [r for r in items if r["label"] is not None]
    random.Random(0).shuffle(items)
    items = items[: 16 if cfg.smoke else 1200]
    llm = make_llm(MODEL_ID, max_model_len=4096, gpu_memory_utilization=0.88)
    tok = llm.get_tokenizer()
    outs = llm.generate([{"prompt_token_ids": render(tok, r["user"], effort=EFFORT)[0]} for r in items], SamplingParams(n=4, temperature=1.0, max_tokens=cfg.max_tokens, seed=0))
    rows = []
    for r, o in zip(items, outs):
        for k, c in enumerate(o.outputs):
            lab = syco_label(tok, r, c.token_ids)
            rows.append({"id": f"{r['id']}#s{k}", "prompt_id": r["id"], "user": r["user"], "effort": EFFORT, "thinking": lab["thinking"], "final": lab["final"], "syco": lab["label"]})
    write_jsonl(cfg.path("syco_natural.jsonl"), rows)
    lab = [r["syco"] for r in rows]
    log(f"syco natural: {len(rows)} samples; sycophantic {lab.count(1)}, not {lab.count(0)}, unparsed {lab.count(None)}")


def stage_syco_attr(cfg, args):
    rows = read_jsonl(cfg.path("syco_natural.jsonl"))
    pairs = pairs_from(rows, "syco", cfg.max_pairs)
    run_attribution(cfg, pairs, "syco_attribution")


STAGES = {
    "natural": (stage_natural, "natural.jsonl"),
    "directions": (stage_directions, "directions.pt"),
    "sweep": (stage_sweep, "sweep_table.json"),
    "sweep_hf": (stage_sweep_hf, "sweep_hf_table.json"),
    "attribution": (stage_attribution, "attribution.json"),
    "latents": (stage_latents, "latents_table.json"),
    "generate": (stage_generate, "synthetic.done"),
    "features": (stage_features, "synthetic.feats.npz"),
    "probes": (stage_probes, "probes_report.json"),
    "syco_natural": (stage_syco_natural, "syco_natural.jsonl"),
    "syco_attr": (stage_syco_attr, "syco_attribution.json"),
    "report": (stage_report, "report.never"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, outputs under data/pools/refusal_smoke")
    ap.add_argument("--force", action="store_true", help="rerun even if the stage output exists")
    args = ap.parse_args()
    cfg = Cfg(args.smoke)
    fn, marker = STAGES[args.stage]
    if cfg.path(marker).exists() and not args.force:
        log(f"{args.stage}: {marker} exists, skipping")
        return
    t0 = time.time()
    fn(cfg, args)
    log(f"stage {args.stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
