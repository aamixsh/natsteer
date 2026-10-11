"""Steering-vector-as-policy testbeds, second pass: refusal (wider latent search, more harmless prompts,
sharper probe evaluation) and sycophancy (steering with directions and attributed latents, exact labels).

Builds on data/pools/refusal_v0 (natural samples, directions, first attribution) written by
refusal_pipeline.py. Everything lands in data/pools/steer_v1 (smoke: steer_v1_smoke); each stage skips
if its output exists. Stages (GPU stages run in separate processes so vLLM and HF never share the GPU):

 refusal
  init              copy the v0 inputs into steer_v1
  natural_more      vLLM: 16 extra samples per mixed refuse/comply prompt (more attribution pairs) and 2
                    samples for 4,000 benign WildChat prompts (more harmless prompts, natural over-refusals)
  attribution_wide  HF: delta-attribution on layers 11, 15, 19, 23 with about 1,000+ pairs
  latents_screen    vLLM: top-32 attributed latents per layer at two strengths, plus activation-difference and
                    random controls (64 prompts each)
  latents_validate  vLLM: survivors on 96 fresh prompts x 2 samples, plus bundles of validated latents
  generate          vLLM: synthetic data from every generator on up to 3,000 harmless prompts
  features          HF: clean pooled features (CoT prefixes of 8/16/32 tokens, CoT, answer) of natural and synthetic rows
  probes            CPU: probes at 8..256 labels, 10 seeds, prompt-clustered bootstrap intervals, controls,
                    precision/recall and prompt-matched discriminator
 sycophancy (exact labels from the model-written-evals answer key)
  syco_directions   HF: difference-of-means directions (end of prompt, CoT, answer) from natural samples
  syco_attr_wide    HF: delta-attribution on layers 11, 15, 19, 23
  syco_sweep        vLLM: direction x layer x positions x coefficient on low-sycophancy items
  syco_latents      vLLM: attributed latents screened on 64 items, validated on 96 others, bundles
  syco_generate     vLLM: synthetic data from every generator on low-sycophancy training items
  syco_features     HF: clean pooled features of natural (train, heldout) and synthetic rows
  syco_probes       CPU: same probe evaluation, natural held-out items
  report            copy the report JSONs to data/runs/steer_v1/<stamp>
Usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/steer_v1.py --stage natural_more [--smoke]
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

from natsteer.config import DATA_ROOT, run_dir  # noqa: E402
from natsteer.harmony import render  # noqa: E402
from natsteer.refusal import compression_ratio, distinct_n, label_generation as refusal_label, load_benign_wildchat, read_jsonl, rep4, write_jsonl  # noqa: E402

POOLS = DATA_ROOT / "pools"
V0 = POOLS / "refusal_v0"
EFFORT = "medium"
LAYERS = (11, 15, 19, 23)
PROMPTED_REFUSAL = "\n\n(Politely decline to help with this request.)"
PROMPTED_SYCO = "\n\n(I feel strongly about the view I described above; please answer in a way that agrees with it.)"


class Cfg:
    def __init__(self, smoke: bool):
        s = self
        s.smoke = smoke
        s.dir = POOLS / ("steer_v1_smoke" if smoke else "steer_v1")
        s.n_wildchat = 40 if smoke else 4000
        s.n_mixed_samples = 4 if smoke else 16
        s.max_mixed_prompts = 10 if smoke else 10**6
        s.max_pairs = 10 if smoke else 1500
        s.top_k = 2 if smoke else 32
        s.screen_lams = (5.0,) if smoke else (2.0, 5.0)
        s.screen_prompts = 8 if smoke else 64
        s.screen_tokens = 384 if smoke else 640
        s.n_controls = 1 if smoke else 4
        s.validate_cap = 3 if smoke else 48
        s.val_prompts = 8 if smoke else 96
        s.val_samples = 1 if smoke else 2
        s.val_tokens = 512 if smoke else 1024
        s.n_data_prompts = 24 if smoke else 3000
        s.gen_tokens = 512 if smoke else 1536
        s.n_random_bundles = 3 if smoke else 40
        s.budgets = (8, 16) if smoke else (8, 16, 32, 64, 256)
        s.seeds = 2 if smoke else 10
        s.n_boot = 20 if smoke else 200
        s.n_jobs = 4 if smoke else 48
        s.syco_select = 8 if smoke else 64
        s.syco_val = 8 if smoke else 96
        s.syco_tokens = 512 if smoke else 1024
        s.syco_coeffs = {"all": (1.0, 4.0) if smoke else (0.5, 1.0, 2.0, 4.0, 8.0), "prompt": (2.0, 8.0) if smoke else (1.0, 2.0, 4.0, 8.0, 16.0)}
        s.syco_top_k = 2 if smoke else 24
        s.syco_data = 24 if smoke else 1500
        s.syco_pairs = 10 if smoke else 600
        s.syco_natural_rows = 40 if smoke else 10**6  # smoke: see syco_natural_rows
        # minimum gain in positive rate for a latent or direction to count as working (smoke: accept anything,
        # so the downstream stages are exercised even though 8 prompts cannot show an effect)
        s.refusal_gain = -1.0 if smoke else 0.2
        s.syco_gain = -1.0 if smoke else 0.15

    def path(self, name):
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / name


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ labelers and summaries


def refusal_labeler(tok, p, gen_ids):
    d = refusal_label(tok, gen_ids)
    d["label"] = d["refusal"]
    return d


def syco_labeler(tok, p, gen_ids):
    from syco_pipeline import label_generation

    d = label_generation(tok, p, gen_ids)  # thinking, final, answer, label, n_gen_tokens
    d["rep4"] = round(rep4(list(gen_ids)), 4)
    return d


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "pos_rate": 0.0, "incomplete_rate": 1.0, "rep4": 1.0, "gen_tokens": 0.0, "final_distinct2": 0.0, "final_compression": 0.0}
    lab = [r["label"] for r in rows]
    finals = [r["final"] for r in rows if r.get("final")]
    return {"n": n, "pos_rate": lab.count(1) / n, "incomplete_rate": lab.count(None) / n, "rep4": float(np.mean([r["rep4"] for r in rows])), "gen_tokens": float(np.mean([r["n_gen_tokens"] for r in rows])), "final_distinct2": distinct_n(finals, 2), "final_compression": compression_ratio(finals)}


def acceptable(s, base, inc=0.15, rep=0.05):
    return s["incomplete_rate"] <= base["incomplete_rate"] + inc and s["rep4"] <= base["rep4"] + rep


META = ("family", "kind", "layer", "positions", "coeff", "latents", "lam", "generator")


def condition_table(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    return {c: summarize(rs) | {k: rs[0].get(k) for k in META} for c, rs in by.items()}


def run_conditions(llm, conditions, prompts, n, max_tokens, path, labeler, keep_text=True, seed=1):
    """conditions: dicts {name, specs, <meta>, optional prompts}; labeled rows are appended to `path` (resumable).
    keep_text=False truncates stored text (screening runs keep metrics, not transcripts)."""
    from vllm import SamplingParams

    from natsteer.vllm_steer import set_steering

    tok = llm.get_tokenizer()
    rows_all = read_jsonl(path)
    done = {r["condition"] for r in rows_all}
    for c in conditions:
        if c["name"] in done:
            continue
        ps = c.get("prompts", prompts)
        if not ps:
            continue
        set_steering(llm, c["specs"])
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": render(tok, p.get("gen_user", p["user"]), effort=EFFORT)[0]} for p in ps], SamplingParams(n=n, temperature=1.0, max_tokens=max_tokens, seed=seed), use_tqdm=False)
        meta = {k: v for k, v in c.items() if k in META}
        rows = []
        for p, o in zip(ps, outs):
            for k, comp in enumerate(o.outputs):
                lab = labeler(tok, p, comp.token_ids)
                r = {"id": f"{p['prompt_id']}|{c['name']}#s{k}", "prompt_id": p["prompt_id"], "user": p["user"], "effort": EFFORT, "condition": c["name"], **meta} | lab
                rows.append(r)
        s = summarize(rows)
        if not keep_text:
            for r in rows:
                r["thinking"], r["final"], r["user"] = None, (r.get("final") or "")[:160], None
        rows_all += rows
        write_jsonl(path, rows_all)
        done.add(c["name"])
        log(f"{c['name']}: pos {s['pos_rate']:.2f} incomplete {s['incomplete_rate']:.2f} rep4 {s['rep4']:.3f} tokens {s['gen_tokens']:.0f} n={s['n']} ({time.time() - t0:.0f}s)")
    set_steering(llm, [])
    return rows_all


_DEC: dict = {}


def decoder(layer):
    """SAE decoder rows [d_sae, 2880] on CPU (cached)."""
    if layer not in _DEC:
        from natsteer.sae import load_andyrdt_sae

        _DEC[layer] = load_andyrdt_sae(layer, 64, device="cpu").W_dec.float().clone()
    return _DEC[layer]


def latent_vec(layer, latent, lam, scale):
    return lam * float(scale[latent]) * decoder(layer)[latent]


def pairs_from(samples, max_pairs, seed=0):
    """Same-prompt pairs (positive sample, negative sample) by `label`."""
    by = defaultdict(lambda: ([], []))
    for s in samples:
        if s.get("label") in (0, 1):
            by[s["prompt_id"]][1 - s["label"]].append(s)
    pairs = [(a, b) for pid in sorted(by) for a, b in zip(*by[pid])]
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]


def run_attribution(cfg, pairs, out_name, with_dirs):
    import torch

    from natsteer.attribution import delta_attribution
    from natsteer.model import load_model
    from natsteer.sae import load_andyrdt_sae

    model, tok = load_model("bf16")
    saes = {L: load_andyrdt_sae(L, 64, device=model.device) for L in LAYERS}
    log(f"attribution on {len(pairs)} pairs, layers {LAYERS}")
    res = delta_attribution(model, tok, pairs, saes, log_every=50)
    dirs = torch.load(V0 / "directions.pt")["dirs"] if with_dirs else None
    summary = {"n_pairs": res["n_pairs"]}
    for L in LAYERS:
        r = res[L]
        top = torch.argsort(r["delta_attr"], descending=True)[:64]
        top_act = torch.argsort(r["delta_act"], descending=True)[:64]
        Wd = saes[L].W_dec.float().cpu()
        e = {
            "top_delta_attr": [{"latent": int(i), "score": float(r["delta_attr"][i]), "t": float(r["delta_attr_t"][i]), "delta_act": float(r["delta_act"][i]), "act_mean_active": float(r["act_mean_active"][i]), "freq": float(r["act_freq"][i])} for i in top],
            "top_delta_act": [{"latent": int(i), "delta_act": float(r["delta_act"][i]), "score": float(r["delta_attr"][i])} for i in top_act],
            "overlap_top20": len(set(top[:20].tolist()) & set(top_act[:20].tolist())),
        }
        if dirs is not None:
            Wn = Wd / Wd.norm(dim=1, keepdim=True)
            for kind in ("eop", "cot"):
                d = dirs[kind][L].float()
                for k in (8, 32, 64):
                    B = Wd[top[:k]].T
                    coef = torch.linalg.lstsq(B, d).solution
                    e[f"r2_{kind}_top{k}"] = float(1 - ((B @ coef - d).norm() / d.norm()) ** 2)
        summary[f"L{L}"] = e
        log(f"L{L}: top attr latents {[x['latent'] for x in e['top_delta_attr'][:10]]}; overlap with delta-act top20 {e['overlap_top20']}")
    torch.save({L: dict(res[L]) for L in LAYERS} | {"n_pairs": res["n_pairs"]}, cfg.path(out_name + ".pt"))
    json.dump(summary, open(cfg.path(out_name + ".json"), "w"), indent=1)


def screen_conditions(att, cfg, top_k, lams, n_controls, seed=0):
    """Single-latent conditions from an attribution result: top-k by delta-attribution, plus
    activation-difference and random-latent controls."""
    import torch

    rng = np.random.default_rng(seed)
    conds = [{"name": "base", "specs": [], "family": "base"}]
    for L in LAYERS:
        r = att[L]
        alive = r["act_freq"] > 1e-5
        score = r["delta_attr"].clone()
        score[~alive] = -1e9
        top = torch.topk(score, top_k).indices.tolist()
        dact = r["delta_act"].clone()
        dact[~alive] = -1e9
        top_act = [i for i in torch.topk(dact, top_k + n_controls).indices.tolist() if i not in top][:n_controls]
        rand = [int(i) for i in rng.choice(np.flatnonzero(alive.numpy()), n_controls, replace=False)]
        for fam, lats, ls in (("attr", top, lams), ("act", top_act, lams[-1:]), ("randlat", rand, lams[-1:])):
            for i in lats:
                for lam in ls:
                    conds.append({"name": f"{fam}_L{L}_{i}_x{lam}", "family": fam, "layer": L, "latents": [i], "lam": lam, "positions": "all", "specs": [{"layer": L, "vec": latent_vec(L, i, lam, r["act_mean_active"]), "positions": "all"}]})
    return conds


def validate_conditions(att, table, cfg, positive_key="pos_rate", min_gain=0.2, cap=None, bundle_lams=(1.0, 2.0)):
    """Survivors of a screen (attr family, positive rate >= base + min_gain, generation intact) at the same
    strength, plus per-layer bundles of survivors weighted by their screening yield."""
    base = table["base"]
    hits = [(c, s) for c, s in table.items() if s.get("family") == "attr" and acceptable(s, base) and s[positive_key] >= base[positive_key] + min_gain]
    hits.sort(key=lambda cs: -cs[1][positive_key])
    hits = hits[: cap or len(hits)]
    conds = [{"name": c, "family": "attr", "layer": s["layer"], "latents": s["latents"], "lam": s["lam"], "positions": "all", "specs": [{"layer": s["layer"], "vec": latent_vec(s["layer"], s["latents"][0], s["lam"], att[s["layer"]]["act_mean_active"]), "positions": "all"}]} for c, s in hits]
    for L in LAYERS:
        ls = [s for _, s in hits if s["layer"] == L]
        for k in (4, 8):
            if len(ls) >= 2:
                sel = ls[:k]
                w = np.array([s[positive_key] for s in sel])
                w = w / w.max()
                for lam in bundle_lams:
                    v = sum(float(wi) * latent_vec(L, s["latents"][0], lam * s["lam"] / max(bundle_lams), att[L]["act_mean_active"]) for wi, s in zip(w, sel))
                    conds.append({"name": f"valbundle_L{L}_k{len(sel)}_x{lam}", "family": "bundle", "layer": L, "latents": [s["latents"][0] for s in sel], "lam": lam, "positions": "all", "specs": [{"layer": L, "vec": v, "positions": "all"}]})
    return conds, [c for c, _ in hits]


# ------------------------------------------------------------------ refusal stages


def load_natural(cfg, include_more=True):
    rows = read_jsonl(cfg.path("natural.jsonl"))
    if include_more:
        rows += read_jsonl(cfg.path("natural_more.jsonl"))
    for r in rows:
        r.setdefault("label", r.get("refusal"))
    if cfg.smoke:  # a small stratified subset: the first 60 prompts of every source, all their samples
        keep = defaultdict(set)
        for r in sorted(rows, key=lambda r: r["prompt_id"]):
            if len(keep[r["source"]]) < 60 or r["prompt_id"] in keep[r["source"]]:
                keep[r["source"]].add(r["prompt_id"])
        rows = [r for r in rows if r["prompt_id"] in keep[r["source"]]]
    return rows


def stage_init(cfg, args):
    for name in ("natural.jsonl", "directions.pt", "selection_prompts.jsonl", "sweep_table.json", "attribution.pt", "attribution.json", "latents_table.json", "syco_natural.jsonl", "syco_attribution.pt", "syco_attribution.json", "generator_choices.json"):
        if (V0 / name).exists() and not cfg.path(name).exists():
            shutil.copy(V0 / name, cfg.path(name))
    cfg.path("init.done").write_text("ok")
    log("init: copied v0 inputs to", cfg.dir)


def stage_natural_more(cfg, args):
    from vllm import SamplingParams

    from natsteer.config import MODEL_ID
    from natsteer.vllm_utils import make_llm

    nat = load_natural(cfg, include_more=False)
    by = defaultdict(list)
    for r in nat:
        by[r["prompt_id"]].append(r)
    mixed = sorted(p for p, rs in by.items() if rs[0]["split"] == "train" and {r["label"] for r in rs} >= {0, 1})[: cfg.max_mixed_prompts]
    mixed_rows = [{k: by[p][0][k] for k in ("prompt_id", "source", "split", "user", "human_label")} for p in mixed]
    wild = load_benign_wildchat(cfg.n_wildchat)
    log(f"{len(mixed_rows)} mixed prompts x {cfg.n_mixed_samples} samples, {len(wild)} WildChat prompts x 2")
    llm = make_llm(MODEL_ID, max_model_len=4096, gpu_memory_utilization=0.88)
    tok = llm.get_tokenizer()
    out = []
    for name, prompts, n, seed in (("m", mixed_rows, cfg.n_mixed_samples, 1), ("w", wild, 2, 2)):
        t0 = time.time()
        gens = llm.generate([{"prompt_token_ids": render(tok, r["user"], effort=EFFORT)[0]} for r in prompts], SamplingParams(n=n, temperature=1.0, max_tokens=1536, seed=seed))
        for r, o in zip(prompts, gens):
            for k, c in enumerate(o.outputs):
                lab = refusal_label(tok, c.token_ids)
                out.append(r | {"id": f"{r['prompt_id']}#{name}{k}", "effort": EFFORT, "finish": c.finish_reason, "label": lab["refusal"]} | lab)
        log(f"{name}: {len(prompts) * n} samples in {time.time() - t0:.0f}s")
    write_jsonl(cfg.path("natural_more.jsonl"), out)
    w = [r for r in out if r["source"] == "wildchat"]
    log(f"WildChat samples {len(w)}: refusals {sum(r['label'] == 1 for r in w)}, incomplete {sum(r['label'] is None for r in w)}")


def stage_attribution_wide(cfg, args):
    samples = [s for s in load_natural(cfg) if s["split"] == "train" and s.get("final")]
    run_attribution(cfg, pairs_from(samples, cfg.max_pairs), "attribution_wide", with_dirs=True)


def split_prompts(cfg):
    """(screening prompts, validation prompts, data prompts): harmless prompts the model always answers,
    from the harmless training set and benign WildChat. Deterministic; the three sets are disjoint."""
    select = read_jsonl(cfg.path("selection_prompts.jsonl"))
    seen = {p["prompt_id"] for p in select}
    by = defaultdict(list)
    for r in load_natural(cfg):
        by[r["prompt_id"]].append(r)
    pool = []
    for pid, rs in sorted(by.items()):
        if pid in seen or rs[0]["source"] not in ("train_harmless", "wildchat") or rs[0]["split"] != "train":
            continue
        if all(r["label"] == 0 for r in rs) and len(rs) >= 2:
            pool.append({"prompt_id": pid, "user": rs[0]["user"], "source": rs[0]["source"]})
    random.Random(0).shuffle(pool)
    return select, pool[: cfg.val_prompts], pool[cfg.val_prompts :][: cfg.n_data_prompts]


def stage_latents_screen(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)  # engine first: CPU torch work before it hangs vLLM's start-up
    att = torch.load(cfg.path("attribution_wide.pt"))
    select, _, _ = split_prompts(cfg)
    conds = screen_conditions(att, cfg, cfg.top_k, cfg.screen_lams, cfg.n_controls)
    log(f"{len(conds)} screening conditions on {cfg.screen_prompts} prompts")
    rows = run_conditions(llm, conds, select[: cfg.screen_prompts], 1, cfg.screen_tokens, cfg.path("latents_screen.jsonl"), refusal_labeler, keep_text=False)
    json.dump(condition_table(rows), open(cfg.path("latents_screen_table.json"), "w"), indent=1)


def stage_latents_validate(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("attribution_wide.pt"))
    table = json.load(open(cfg.path("latents_screen_table.json")))
    _, val, _ = split_prompts(cfg)
    conds, names = validate_conditions(att, table, cfg, min_gain=cfg.refusal_gain, cap=cfg.validate_cap)
    log(f"{len(names)} screening hits; {len(conds)} validation conditions on {len(val)} fresh prompts")
    base_cond = {"name": "base", "specs": [], "family": "base"}
    rows = run_conditions(llm, [base_cond] + conds, val, cfg.val_samples, cfg.val_tokens, cfg.path("latents_validated.jsonl"), refusal_labeler)
    t = condition_table(rows)
    json.dump(t, open(cfg.path("latents_validated_table.json"), "w"), indent=1)
    base = t["base"]
    good = [c for c, s in t.items() if s.get("family") == "attr" and acceptable(s, base) and s["pos_rate"] >= base["pos_rate"] + cfg.refusal_gain]
    log(f"validated single latents (>=20% on fresh prompts, generation intact): {len(good)} of {len(names)}")


def decide_generators(cfg, task):
    """Pick the generators for synthetic data from the validation tables."""
    import torch

    out = {}
    if task == "refusal":
        t = json.load(open(cfg.path("latents_validated_table.json")))
        sweep = json.load(open(cfg.path("sweep_table.json")))
        base_s = sweep["base"]
        sweep_ok = [(c, s) for c, s in sweep.items() if s.get("family") == "dom" and acceptable(s, base_s) and s["refusal_rate"] >= base_s["refusal_rate"] + cfg.refusal_gain]
        for pos in ("all", "prompt"):
            cand = [(c, s) for c, s in sweep_ok if s["positions"] == pos]
            if cand:
                out["dom_" + pos] = max(cand, key=lambda cs: cs[1]["refusal_rate"])[0]
    else:
        t = json.load(open(cfg.path("syco_latents_validated_table.json")))
        sweep = json.load(open(cfg.path("syco_sweep_table.json")))
        base_s = sweep["base"]
        sweep_ok = [(c, s) for c, s in sweep.items() if s.get("family") == "dom" and acceptable(s, base_s) and s["pos_rate"] >= base_s["pos_rate"] + cfg.syco_gain]
        for pos in ("all", "prompt"):
            cand = [(c, s) for c, s in sweep_ok if s["positions"] == pos]
            if cand:
                out["dom_" + pos] = max(cand, key=lambda cs: cs[1]["pos_rate"])[0]
    base = t["base"]
    gain = cfg.refusal_gain if task == "refusal" else cfg.syco_gain
    out["latents"] = sorted([c for c, s in t.items() if s.get("family") == "attr" and acceptable(s, base) and s["pos_rate"] >= base["pos_rate"] + gain], key=lambda c: -t[c]["pos_rate"])
    bund = [(c, s) for c, s in t.items() if s.get("family") == "bundle" and acceptable(s, base) and s["pos_rate"] >= base["pos_rate"] + gain]
    if bund:
        out["valbundle"] = max(bund, key=lambda cs: cs[1]["pos_rate"])[0]
    return out, t


def partition(prompts, k):
    return [prompts[i::k] for i in range(k)]


def build_generators(cfg, task, att, choices, vt, sweep, data, prompted_suffix):
    """Generator conditions: dense directions, a validated bundle, one validated latent per prompt group
    ('latent_mix'), random bundles of validated latents ('random_bundles'), prompted, random direction."""
    import torch

    D = torch.load(cfg.path("directions.pt" if task == "refusal" else "directions_syco.pt"))["dirs"]
    gens = []

    def dom_spec(c):
        s = sweep[c]
        return [{"layer": s["layer"], "vec": s["coeff"] * D[s["kind"]][s["layer"]], "positions": s["positions"]}]

    for pos in ("all", "prompt"):
        if "dom_" + pos in choices:
            gens.append({"name": "dom_" + pos, "generator": "dom_" + pos, "specs": dom_spec(choices["dom_" + pos]), "prompts": data})
    if "dom_all" in choices:
        s = sweep[choices["dom_all"]]
        g = torch.randn(2880, generator=torch.Generator().manual_seed(0))
        gens.append({"name": "random_dir", "generator": "random_dir", "specs": [{"layer": s["layer"], "vec": g / g.norm() * float((s["coeff"] * D[s["kind"]][s["layer"]]).norm()), "positions": "all"}], "prompts": data})
    if "valbundle" in choices:
        gens.append({"name": "valbundle", "generator": "valbundle", "specs": [{"layer": vt[choices["valbundle"]]["layer"], "vec": None, "positions": "all"}], "prompts": data, "_from": choices["valbundle"]})
    lat = choices.get("latents", [])
    if lat:
        for c, part in zip(lat, partition(data, len(lat))):
            s = vt[c]
            gens.append({"name": f"latent_mix:{c}", "generator": "latent_mix", "specs": [{"layer": s["layer"], "vec": latent_vec(s["layer"], s["latents"][0], s["lam"], att[s["layer"]]["act_mean_active"]), "positions": "all"}], "prompts": part})
    if len(lat) >= 3:
        rng = random.Random(0)
        for b, part in enumerate(partition(data, cfg.n_random_bundles)):
            pick = rng.sample(lat, 3)
            by_layer = defaultdict(lambda: 0)
            for c in pick:
                s = vt[c]
                w = rng.uniform(0.5, 1.0) * 0.6
                by_layer[s["layer"]] = by_layer[s["layer"]] + w * latent_vec(s["layer"], s["latents"][0], s["lam"], att[s["layer"]]["act_mean_active"])
            gens.append({"name": f"random_bundles:{b}", "generator": "random_bundles", "specs": [{"layer": L, "vec": v, "positions": "all"} for L, v in by_layer.items()], "prompts": part})
    gens.append({"name": "prompted", "generator": "prompted", "specs": [], "prompts": [p | {"gen_user": p["user"] + prompted_suffix} for p in data]})
    return gens


def stage_generate(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("attribution_wide.pt"))
    _, _, data = split_prompts(cfg)
    choices, vt = decide_generators(cfg, "refusal")
    sweep = json.load(open(cfg.path("sweep_table.json")))
    log("generators:", json.dumps(choices)[:600])
    json.dump(choices, open(cfg.path("generator_choices_v1.json"), "w"), indent=1)
    gens = build_generators(cfg, "refusal", att, choices, vt, sweep, data, PROMPTED_REFUSAL)
    for g in gens:  # the validated bundle's vector is rebuilt from the screening table, as in the validation stage
        if g.get("_from"):
            conds, _ = validate_conditions(att, json.load(open(cfg.path("latents_screen_table.json"))), cfg, min_gain=cfg.refusal_gain, cap=cfg.validate_cap)
            g["specs"] = next(c["specs"] for c in conds if c["name"] == g["_from"])
    run_conditions(llm, gens, data, 1, cfg.gen_tokens, cfg.path("synthetic.jsonl"), refusal_labeler)
    cfg.path("synthetic.done").write_text("ok")


FEAT_LAYERS = (11, 15, 19)
FEAT_POOLINGS = ("eop_mean", "cot_mean", "cot_first8", "cot_first16", "cot_first32", "final_mean")


def extract_to(cfg, model, tok, rows, name, layers, poolings):
    from natsteer.features import extract, save

    rows = [r for r in rows if r.get("label") is not None]
    t0 = time.time()
    save(cfg.path(f"{name}.feats.npz"), rows, extract(model, tok, rows, layers, poolings, max_len=3072, log_every=1500))
    log(f"features {name}: {len(rows)} rows in {time.time() - t0:.0f}s")


def stage_features(cfg, args):
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    for name, rows in (("natural", load_natural(cfg)), ("synthetic", read_jsonl(cfg.path("synthetic.jsonl")))):
        if not cfg.path(f"{name}.feats.npz").exists():
            extract_to(cfg, model, tok, rows, name, FEAT_LAYERS, FEAT_POOLINGS)


# ------------------------------------------------------------------ probe evaluation (shared)


def load_feats(cfg, name, rows_name, layers_keys=None):
    rows = [r for r in (read_jsonl(cfg.path(rows_name)) if isinstance(rows_name, str) else rows_name) if r.get("label") is not None]
    f = np.load(cfg.path(f"{name}.feats.npz"))
    assert list(f["ids"]) == [r["id"] for r in rows], name
    return rows, f


def evaluate_task(cfg, task, nat_rows, nat_f, syn_rows, syn_f, keys, dirs, neg_mask, train_mask, eval_mask, slices_fn, groups_ev, control_keys, prompt_key, report_name):
    """Shared probe evaluation. nat_*: natural rows and features; syn_*: synthetic rows (label 1 only used) and features.
    neg_mask: natural negatives on the synthetic data prompts; train_mask/eval_mask: natural training and held-out rows."""
    from natsteer.probe_eval import matched_discriminator, precision_recall, run_arm_grid

    yn = np.array([r["label"] for r in nat_rows])
    ys = np.array([r["label"] for r in syn_rows])
    gen = np.array([r["generator"] for r in syn_rows])
    gens = sorted(set(gen))
    tr_idx, ev_idx = np.flatnonzero(train_mask), np.flatnonzero(eval_mask)
    neg_idx = np.flatnonzero(neg_mask)
    pos_by_gen = {g: np.flatnonzero((gen == g) & (ys == 1)) for g in gens}
    report = {"n_natural_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)), "n_natural_neg_on_data_prompts": int(len(neg_idx)), "generators": {g: {"n": int((gen == g).sum()), "verified_pos": int(len(pos_by_gen[g]))} for g in gens}, "features": {}, "controls": {}, "coverage": {}}
    slices = {k: v for k, v in slices_fn(nat_rows, ev_idx).items() if len(v) >= 20}
    gev = groups_ev
    for key in keys:
        Xn, Xs = nat_f[key].astype(np.float32), syn_f[key].astype(np.float32)
        ok_n, ok_s = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
        tr_k, ev_k, neg_k = tr_idx[ok_n[tr_idx]], ev_idx[ok_n[ev_idx]], neg_idx[ok_n[neg_idx]]
        if len(ev_k) == 0 or len(tr_k) == 0:
            raise ValueError(f"{key}: no natural {'held-out' if len(ev_k) == 0 else 'training'} rows with features; check the split masks")
        pos_k = {g: p[ok_s[p]] for g, p in pos_by_gen.items()}
        trp, trn = tr_k[yn[tr_k] == 1], tr_k[yn[tr_k] == 0]

        def natural(b, seed, trp=trp, trn=trn):
            rng = np.random.default_rng(seed)
            if min(len(trp), len(trn)) < b // 2:
                return None
            i = np.r_[rng.choice(trp, b // 2, replace=False), rng.choice(trn, b // 2, replace=False)]
            return Xn[i], yn[i]

        def synthetic(g, frac_nat, pos_k=pos_k, neg_k=neg_k, trp=trp, trn=trn):
            def fn(b, seed):
                rng = np.random.default_rng(100 + seed)
                nat_each = int(b * frac_nat / 2)
                syn_pos = b // 2 - nat_each
                if len(pos_k[g]) < syn_pos or len(neg_k) < b // 2 - nat_each or min(len(trp), len(trn)) < nat_each:
                    return None
                i_s = rng.choice(pos_k[g], syn_pos, replace=False)
                i_neg = rng.choice(neg_k, b // 2 - nat_each, replace=False)
                parts_X, parts_y = [Xs[i_s], Xn[i_neg]], [np.ones(len(i_s)), np.zeros(len(i_neg))]
                if nat_each:
                    j = np.r_[rng.choice(trp, nat_each, replace=False), rng.choice(trn, nat_each, replace=False)]
                    parts_X.append(Xn[j])
                    parts_y.append(yn[j])
                return np.concatenate(parts_X), np.concatenate(parts_y).astype(int)

            return fn

        arms = {"natural": natural}
        for g in gens:
            if g == "random_dir":
                continue
            arms[f"synthetic_{g}"] = synthetic(g, 0.0)
            arms[f"mix_{g}"] = synthetic(g, 0.5)
        res = run_arm_grid(arms, Xn[ev_k], yn[ev_k], gev[ev_k], {k: np.isin(ev_k, v) for k, v in slices.items()}, cfg.budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs)
        report["features"][key] = res
        log(f"{key}: probes done")
        if key in control_keys:  # projection controls at the largest budget
            L = int(key.split("_")[0][1:])
            ctl = {}
            rng = np.random.default_rng(0)
            sub = {a: arms[a] for a in ("natural", "synthetic_dom_all", "synthetic_latent_mix", "synthetic_prompted") if a in arms}
            projs = {f"dir_{k}": (dirs[k][L].float().numpy() / np.linalg.norm(dirs[k][L].float().numpy())) for k in dirs if k != "eop_dataset"}
            for r in range(2):
                v = rng.standard_normal(Xn.shape[1])
                projs[f"random{r}"] = v / np.linalg.norm(v)
            for pname, u in projs.items():
                out = run_arm_grid(sub, Xn[ev_k], yn[ev_k], gev[ev_k], {"all": np.ones(len(ev_k), bool)}, (cfg.budgets[-1],), cfg.seeds, n_boot=max(20, cfg.n_boot // 4), n_jobs=cfg.n_jobs, project=u.astype(np.float32))
                ctl[pname] = out["all"][str(cfg.budgets[-1])]
            report["controls"][key] = ctl
        if key in control_keys or key.endswith("cot_first32"):  # coverage and matched discriminator
            nat_pos = tr_k[yn[tr_k] == 1]
            cov = {}
            Pn_key = prompt_key
            Pn, Ps = nat_f[Pn_key].astype(np.float32), syn_f[Pn_key].astype(np.float32)
            gn = np.array([r["prompt_id"] for r in nat_rows])
            gs = np.array([r["prompt_id"] for r in syn_rows])
            for g in gens:
                p = pos_k[g]
                if len(p) < (8 if cfg.smoke else 30) or len(nat_pos) < 30:
                    continue
                p = p[~np.isnan(Ps[p]).any(1)]
                npos = nat_pos[~np.isnan(Pn[nat_pos]).any(1)]
                cov[g] = precision_recall(Xn[nat_pos], Xs[p]) | {"matched": matched_discriminator(Xs[p], gs[p], Ps[p], Xn[npos], gn[npos], Pn[npos])}
            report["coverage"][key] = cov
    return report


def text_diversity(rows_by_name):
    out = {}
    for name, rows in rows_by_name.items():
        rows = [r for r in rows if r.get("final")][:1000]
        out[name] = {"n": len(rows), "final_compression": compression_ratio(r["final"] for r in rows), "final_distinct2": distinct_n([r["final"] for r in rows], 2), "cot_compression": compression_ratio(r["thinking"] for r in rows), "cot_distinct2": distinct_n([r["thinking"] for r in rows], 2)}
    return out


def stage_probes(cfg, args):
    import torch

    nat_rows, nat_f = load_feats(cfg, "natural", load_natural(cfg))
    syn_rows, syn_f = load_feats(cfg, "synthetic", "synthetic.jsonl")
    _, _, data = split_prompts(cfg)
    data_ids = {p["prompt_id"] for p in data}
    split = np.array([r["split"] for r in nat_rows])
    src = np.array([r["source"] for r in nat_rows])
    hl = np.array([r.get("human_label") if r.get("human_label") is not None else -1 for r in nat_rows])
    pid = np.array([r["prompt_id"] for r in nat_rows])
    yn = np.array([r["label"] for r in nat_rows])

    def slices_fn(rows, ev_idx):
        e = np.zeros(len(rows), bool)
        e[ev_idx] = True
        return {"all": ev_idx, "harmful_prompts": np.flatnonzero(e & (np.isin(src, ["test_harmful"]) | ((src == "xstest") & (hl == 1)))), "harmless_prompts": np.flatnonzero(e & (np.isin(src, ["test_harmless"]) | ((src == "xstest") & (hl == 0)))), "xstest": np.flatnonzero(e & (src == "xstest"))}

    dirs = torch.load(cfg.path("directions.pt"))["dirs"]
    train_mask = split == "train"
    neg_mask = np.isin(pid, list(data_ids)) & (yn == 0)
    keys = [f"L15_cot_first{n}" for n in (8, 16, 32)] + ["L15_cot_mean", "L19_cot_first32", "L11_cot_first32", "L15_final_mean", "L15_eop_mean"]
    report = evaluate_task(cfg, "refusal", nat_rows, nat_f, syn_rows, syn_f, keys, dirs, neg_mask, train_mask, split == "eval", slices_fn, pid, ("L15_cot_first32", "L15_cot_mean"), "L15_eop_mean", "probes_report")
    report["diversity"] = text_diversity({**{g: [r for r in syn_rows if r["generator"] == g and r["label"] == 1] for g in sorted({r["generator"] for r in syn_rows})}, "natural_refusals": [r for r in nat_rows if r["label"] == 1 and r["split"] == "train"][:1000]})
    json.dump(report, open(cfg.path("probes_report.json"), "w"), indent=1)


# ------------------------------------------------------------------ sycophancy stages


def syco_state(cfg):
    """Items for steering: low-sycophancy items (natural rate <= 25% over >= 3 parsed samples), split into
    screening, validation and data items; prompt rows carry the answer key for the oracle."""
    train = {r["id"]: r for r in read_jsonl(POOLS / "syco_v0" / "train.jsonl")}
    per = defaultdict(list)
    for r in read_jsonl(cfg.path("syco_natural.jsonl")):
        per[r["prompt_id"]].append(r["syco"])
    rate = {p: float(np.mean([x for x in l if x is not None])) for p, l in per.items() if sum(x is not None for x in l) >= 3}
    low = sorted(p for p, v in rate.items() if v <= 0.25)
    random.Random(0).shuffle(low)
    select, val = low[: cfg.syco_select], low[cfg.syco_select : cfg.syco_select + cfg.syco_val]
    used = set(select) | set(val)
    data = sorted(i for i, r in train.items() if r["label"] == 0 and i not in used)
    random.Random(1).shuffle(data)
    mk = lambda ids: [{"prompt_id": i, "user": train[i]["user"], "matching": train[i]["matching"], "not_matching": train[i]["not_matching"], "source": train[i]["source"]} for i in ids]  # noqa: E731
    return mk(select), mk(val), mk(data[: cfg.syco_data]), rate


def stage_syco_directions(cfg, args):
    import torch

    from natsteer.features import extract
    from natsteer.model import load_model

    rows = read_jsonl(cfg.path("syco_natural.jsonl"))[: cfg.syco_natural_rows]
    for r in rows:
        r["label"] = r["syco"]
    rng = random.Random(0)
    by = defaultdict(list)
    for r in rows:
        by[r["prompt_id"]].append(r)
    rate = {p: np.mean([x["label"] for x in rs if x["label"] is not None]) for p, rs in by.items() if sum(x["label"] is not None for x in rs) >= 3}
    hi, lo = [p for p, v in rate.items() if v >= 0.75], [p for p, v in rate.items() if v <= 0.25]
    model, tok = load_model("bf16")
    dirs = {}

    def mean_feat(rs, pooling):
        f = extract(model, tok, rs, LAYERS, (pooling,), max_len=2048, log_every=0)
        return np.stack([np.nanmean(f[L][pooling].astype(np.float32), 0) for L in LAYERS])

    def full(arr):
        out = torch.zeros(24, 2880)
        for i, L in enumerate(LAYERS):
            out[L] = torch.tensor(arr[i])
        return out

    first = lambda ps: [by[p][0] for p in ps]  # noqa: E731
    dirs["eop"] = full(mean_feat(first(hi), "eop_mean") - mean_feat(first(lo), "eop_mean"))
    for kind, pooling in (("cot", "cot_mean"), ("final", "final_mean")):
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        pos, neg = rng.sample(pos, min(len(pos), 800)), rng.sample(neg, min(len(neg), 800))
        dirs[kind] = full(mean_feat(pos, pooling) - mean_feat(neg, pooling))
    info = {"n_hi": len(hi), "n_lo": len(lo), "norm": {k: {L: float(v[L].norm()) for L in LAYERS} for k, v in dirs.items()}}
    torch.save({"dirs": dirs, "info": info}, cfg.path("directions_syco.pt"))
    log("syco directions:", json.dumps(info))


def stage_syco_attr_wide(cfg, args):
    rows = read_jsonl(cfg.path("syco_natural.jsonl"))
    for r in rows:
        r["label"] = r["syco"]
    run_attribution(cfg, pairs_from(rows, cfg.syco_pairs), "syco_attribution_wide", with_dirs=False)


def stage_syco_sweep(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    select, _, _, _ = syco_state(cfg)
    D = torch.load(cfg.path("directions_syco.pt"))["dirs"]
    g = torch.randn(2880, generator=torch.Generator().manual_seed(0))
    conds = [{"name": "base", "specs": [], "family": "base"}]
    for kind in ("eop", "cot", "final"):
        for L in (11, 15, 19):
            for pos in ("all", "prompt"):
                for c in cfg.syco_coeffs[pos]:
                    conds.append({"name": f"{kind}_L{L}_{pos}_c{c}", "family": "dom", "kind": kind, "layer": L, "positions": pos, "coeff": c, "specs": [{"layer": L, "vec": c * D[kind][L], "positions": pos}]})
    for pos in ("all", "prompt"):
        for c in sorted({cfg.syco_coeffs[pos][1], cfg.syco_coeffs[pos][-1]}):
            conds.append({"name": f"random_L15_{pos}_c{c}", "family": "random", "kind": "random", "layer": 15, "positions": pos, "coeff": c, "specs": [{"layer": 15, "vec": c * float(D["final"][15].norm()) * g / g.norm(), "positions": pos}]})
    log(f"{len(conds)} sweep conditions on {len(select)} low-sycophancy items")
    rows = run_conditions(llm, conds, select, 1, cfg.syco_tokens, cfg.path("syco_sweep.jsonl"), syco_labeler)
    json.dump(condition_table(rows), open(cfg.path("syco_sweep_table.json"), "w"), indent=1)


def stage_syco_latents(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("syco_attribution_wide.pt"))
    select, val, _, _ = syco_state(cfg)
    conds = screen_conditions(att, cfg, cfg.syco_top_k, cfg.screen_lams, cfg.n_controls)
    log(f"{len(conds)} screening conditions on {len(select)} items")
    rows = run_conditions(llm, conds, select, 1, cfg.syco_tokens, cfg.path("syco_latents_screen.jsonl"), syco_labeler, keep_text=False)
    table = condition_table(rows)
    json.dump(table, open(cfg.path("syco_latents_screen_table.json"), "w"), indent=1)
    vconds, names = validate_conditions(att, table, cfg, min_gain=cfg.syco_gain, cap=cfg.validate_cap)
    log(f"{len(names)} screening hits; {len(vconds)} validation conditions on {len(val)} fresh items")
    rows = run_conditions(llm, [{"name": "base", "specs": [], "family": "base"}] + vconds, val, 2 if not cfg.smoke else 1, cfg.syco_tokens, cfg.path("syco_latents_validated.jsonl"), syco_labeler)
    json.dump(condition_table(rows), open(cfg.path("syco_latents_validated_table.json"), "w"), indent=1)


def stage_syco_generate(cfg, args):
    import torch

    from natsteer.vllm_steer import make_steerable_llm

    llm = make_steerable_llm(gpu_memory_utilization=0.88)
    att = torch.load(cfg.path("syco_attribution_wide.pt"))
    _, _, data, _ = syco_state(cfg)
    choices, vt = decide_generators(cfg, "syco")
    sweep = json.load(open(cfg.path("syco_sweep_table.json")))
    json.dump(choices, open(cfg.path("syco_generator_choices.json"), "w"), indent=1)
    log("generators:", json.dumps(choices)[:600])
    gens = build_generators(cfg, "syco", att, choices, vt, sweep, data, PROMPTED_SYCO)
    for g in gens:
        if g.get("_from"):
            table = json.load(open(cfg.path("syco_latents_screen_table.json")))
            conds, _ = validate_conditions(att, table, cfg, min_gain=cfg.syco_gain, cap=cfg.validate_cap)
            g["specs"] = next(c["specs"] for c in conds if c["name"] == g["_from"])
    run_conditions(llm, gens, data, 1, cfg.syco_tokens, cfg.path("syco_synthetic.jsonl"), syco_labeler)
    cfg.path("syco_synthetic.done").write_text("ok")


def syco_natural_rows(cfg):
    out = []
    for name in ("train", "heldout"):
        for r in read_jsonl(POOLS / "syco_v0" / f"{name}.jsonl"):
            if r["label"] is not None:
                out.append({k: r[k] for k in ("id", "source", "user", "thinking", "final", "effort", "label")} | {"prompt_id": r["id"], "split": name})
    if cfg.smoke:  # the data items (their natural negatives are needed), plus a few train and held-out rows
        data_ids = {p["prompt_id"] for p in syco_state(cfg)[2]}
        train = [r for r in out if r["split"] == "train"]
        keep = [r for r in train if r["id"] in data_ids] + [r for r in train if r["id"] not in data_ids][:60] + [r for r in out if r["split"] == "heldout"][:80]
        return keep
    return out


SYCO_LAYERS = (11, 15, 19)
SYCO_POOLINGS = ("prompt_last", "cot_first8", "cot_first32", "cot_mean", "final_mean")


def stage_syco_features(cfg, args):
    from natsteer.model import load_model

    model, tok = load_model("bf16")
    for name, rows in (("syco_natural", syco_natural_rows(cfg)), ("syco_synthetic", read_jsonl(cfg.path("syco_synthetic.jsonl")))):
        if not cfg.path(f"{name}.feats.npz").exists():
            extract_to(cfg, model, tok, rows, name, SYCO_LAYERS, SYCO_POOLINGS)


def stage_syco_probes(cfg, args):
    import torch

    nat_rows, nat_f = load_feats(cfg, "syco_natural", syco_natural_rows(cfg))
    syn_rows, syn_f = load_feats(cfg, "syco_synthetic", "syco_synthetic.jsonl")
    _, _, data, _ = syco_state(cfg)
    data_ids = {p["prompt_id"] for p in data}
    split = np.array([r["split"] for r in nat_rows])
    src = np.array([r["source"] for r in nat_rows])
    pid = np.array([r["prompt_id"] for r in nat_rows])
    yn = np.array([r["label"] for r in nat_rows])

    def slices_fn(rows, ev_idx):
        return {"all": ev_idx, "nlp_survey": ev_idx[src[ev_idx] == "nlp_survey"], "political_typology": ev_idx[src[ev_idx] == "political_typology_quiz"]}

    dirs = torch.load(cfg.path("directions_syco.pt"))["dirs"]
    keys = ["L15_final_mean", "L15_cot_first32", "L15_cot_mean", "L19_final_mean", "L11_final_mean", "L15_prompt_last"]
    report = evaluate_task(cfg, "syco", nat_rows, nat_f, syn_rows, syn_f, keys, dirs, np.isin(pid, list(data_ids)) & (yn == 0) & (split == "train"), split == "train", split == "heldout", slices_fn, pid, ("L15_final_mean",), "L15_prompt_last", "syco_probes_report")
    report["diversity"] = text_diversity({g: [r for r in syn_rows if r["generator"] == g and r["label"] == 1] for g in sorted({r["generator"] for r in syn_rows})} | {"natural_sycophantic": [r for r in nat_rows if r["label"] == 1 and r["split"] == "train"][:1000]})
    json.dump(report, open(cfg.path("syco_probes_report.json"), "w"), indent=1)


def stage_report(cfg, args):
    out = run_dir("steer_v1_smoke" if cfg.smoke else "steer_v1")
    for p in sorted(cfg.dir.glob("*.json")):
        shutil.copy(p, out / p.name)
    (out / "report.done").write_text("ok")
    log("saved", out)


STAGES = {
    "init": (stage_init, "init.done"),
    "natural_more": (stage_natural_more, "natural_more.jsonl"),
    "attribution_wide": (stage_attribution_wide, "attribution_wide.json"),
    "latents_screen": (stage_latents_screen, "latents_screen_table.json"),
    "latents_validate": (stage_latents_validate, "latents_validated_table.json"),
    "generate": (stage_generate, "synthetic.done"),
    "features": (stage_features, "synthetic.feats.npz"),
    "probes": (stage_probes, "probes_report.json"),
    "syco_directions": (stage_syco_directions, "directions_syco.pt"),
    "syco_attr_wide": (stage_syco_attr_wide, "syco_attribution_wide.json"),
    "syco_sweep": (stage_syco_sweep, "syco_sweep_table.json"),
    "syco_latents": (stage_syco_latents, "syco_latents_validated_table.json"),
    "syco_generate": (stage_syco_generate, "syco_synthetic.done"),
    "syco_features": (stage_syco_features, "syco_synthetic.feats.npz"),
    "syco_probes": (stage_syco_probes, "syco_probes_report.json"),
    "report": (stage_report, "report.never"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, outputs under data/pools/steer_v1_smoke")
    ap.add_argument("--force", action="store_true", help="rerun even if the stage output exists")
    args = ap.parse_args()
    cfg = Cfg(args.smoke)
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
