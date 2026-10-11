"""Extras that run after steer_v1.py (same data/pools/steer_v1 directory); they only read its outputs.

  laser_warmstart   CPU: LASER and random sampling on a rare-positive pool of benign WildChat traffic (natural
                    refusals are about 4.5% of samples), with and without free synthetic labeled examples as
                    a warm start; evaluated on held-out WildChat prompts. How many natural labels do steered
                    positives save?
  probes_wildchat   CPU: probes trained on natural data (any source, or WildChat only) versus synthetic positives
                    versus mixes, evaluated on held-out WildChat prompts (natural over-refusal on benign
                    traffic), with the same clustered intervals as steer_v1 probes.
  reach             GPU: reachability of steered activations. For every validated latent and every generator:
                    SAE reconstruction error, perturbation size and nearest-neighbor distance to a bank of
                    natural activations, steered versus unsteered text, against the natural floor, next to
                    the condition's refusal yield.
  timing_hf         GPU: steering timing for validated latents and the dense direction: prompt tokens only,
                    CoT tokens only, answer tokens only (HF steering, 32 prompts each).
  summary           write SUMMARY.md next to the outputs (scripts/morning_summary.py): one readable page
  report_extra      copy the report JSONs and SUMMARY.md to data/runs/steer_v1_extra/<stamp>
Usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/steer_v1_extra.py --stage reach [--smoke]
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import steer_v1 as sv  # noqa: E402
from natsteer.config import run_dir  # noqa: E402
from natsteer.harmony import render  # noqa: E402
from natsteer.refusal import read_jsonl, write_jsonl  # noqa: E402

log = sv.log


class XCfg(sv.Cfg):
    def __init__(self, smoke):
        super().__init__(smoke)
        self.reach_rows = 6 if smoke else 24
        self.reach_bank_tokens = 3000 if smoke else 60000
        self.reach_nn_queries = 500 if smoke else 4096
        self.timing_sets = 2 if smoke else 8
        self.timing_prompts = 4 if smoke else 32
        self.timing_tokens = 256 if smoke else 768
        self.laser_budget = 48 if smoke else 256
        self.laser_batch = 8 if smoke else 16
        self.laser_seeds = 2 if smoke else 5
        self.laser_extra = 16 if smoke else 64
        self.wild_budgets = (8, 16) if smoke else (8, 16, 32, 64, 128)


def wild_heldout(pid: str) -> bool:
    """Stable 30% of prompts held out for the benign-traffic evaluation."""
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 10 < 3


# ------------------------------------------------------------------ CPU: LASER warm start


def _wild_data(cfg):
    nat_rows, nat_f = sv.load_feats(cfg, "natural", sv.load_natural(cfg))
    syn_rows, syn_f = sv.load_feats(cfg, "synthetic", "synthetic.jsonl")
    pid = np.array([r["prompt_id"] for r in nat_rows])
    held = np.array([wild_heldout(p) for p in pid])
    wild = np.array([r["source"] == "wildchat" for r in nat_rows])
    y = np.array([r["label"] for r in nat_rows])
    _, _, data = sv.split_prompts(cfg)
    data_ids = {p["prompt_id"] for p in data}
    spid = np.array([r["prompt_id"] for r in syn_rows])
    syn_ok = ~np.array([wild_heldout(p) for p in spid])
    return nat_rows, nat_f, syn_rows, syn_f, pid, held, wild, y, data_ids, syn_ok


def _laser_job(args):
    from natsteer import laser

    arm, seed = args
    c = _LASER_CTX
    extra = None
    if "+" in arm:
        base, g = arm.split("+")
        rng = np.random.default_rng(1000 + seed)
        pos, neg = c["syn_pos"][g], c["neg"]
        k = c["n_extra"]
        if len(pos) < k or len(neg) < k:
            return args, None
        i, j = rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)
        extra = (np.concatenate([c["Xs"][i], c["Xn"][j]]), np.r_[np.ones(k), np.zeros(k)].astype(int))
    else:
        base = arm
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        curve = laser.run(c["Xp"], lambda idx: c["yp"][idx], c["Xe"], c["ye"], budget=c["budget"], batch=c["batch"], init=c["batch"], strategy=base, embed_dim=64, seed=seed, extra=extra)
    return args, curve


_LASER_CTX: dict = {}


def stage_laser_warmstart(cfg, args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    nat_rows, nat_f, syn_rows, syn_f, pid, held, wild, y, data_ids, syn_ok = _wild_data(cfg)
    gen = np.array([r["generator"] for r in syn_rows])
    ys = np.array([r["label"] for r in syn_rows])
    out = {"pool": {}, "keys": {}}
    for key in ("L15_cot_first32", "L15_cot_mean"):
        Xn, Xs = nat_f[key].astype(np.float32), syn_f[key].astype(np.float32)
        ok_n, ok_s = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
        pool, ev = np.flatnonzero(wild & ~held & ok_n), np.flatnonzero(wild & held & ok_n)
        neg = np.flatnonzero((np.isin(pid, list(data_ids))) & ~held & (y == 0) & ok_n)
        gens = [g for g in ("dom_all", "latent_mix", "random_bundles", "valbundle", "prompted") if g in set(gen)]
        syn_pos = {g: np.flatnonzero((gen == g) & (ys == 1) & syn_ok & ok_s) for g in gens}
        _LASER_CTX.clear()
        _LASER_CTX.update(Xp=Xn[pool], yp=y[pool], Xe=Xn[ev], ye=y[ev], Xs=Xs, Xn=Xn, syn_pos=syn_pos, neg=neg, n_extra=cfg.laser_extra, budget=cfg.laser_budget, batch=cfg.laser_batch)
        out["pool"] = {"n_pool": int(len(pool)), "pool_pos_rate": float(y[pool].mean()), "n_eval": int(len(ev)), "eval_pos": int(y[ev].sum())}
        arms = ["laser", "random"] + [f"{b}+{g}" for g in gens for b in ("laser", "random")]
        jobs = [(a, s) for a in arms for s in range(cfg.laser_seeds)]
        res = {}
        with ProcessPoolExecutor(max_workers=min(cfg.n_jobs, len(jobs)), mp_context=multiprocessing.get_context("fork")) as ex:
            for a, curve in ex.map(_laser_job, jobs):
                if curve is not None:
                    res.setdefault(a[0], []).append(curve)
        summary = {}
        for a, curves in res.items():
            n_pts = min(len(c) for c in curves)
            rows = []
            for i in range(n_pts):
                pts = [c[i] for c in curves]
                row = {"n_labels": pts[0]["n_labels"], "n_pos": float(np.mean([p["n_pos"] for p in pts])), "n_seeds": len(pts)}
                for m in ("auroc", "tpr@0.01", "tpr@0.05"):
                    v = [p[m] for p in pts if m in p]
                    if v:
                        row[m] = {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                rows.append(row)
            summary[a] = rows
        out["keys"][key] = summary
        log(f"{key}: laser warm start done ({len(arms)} arms, pool {len(pool)}, {int(y[pool].sum())} positives)")
    json.dump(out, open(cfg.path("laser_warmstart.json"), "w"), indent=1)


# ------------------------------------------------------------------ CPU: probes on held-out WildChat


def stage_probes_wildchat(cfg, args):
    from natsteer.probe_eval import run_arm_grid

    nat_rows, nat_f, syn_rows, syn_f, pid, held, wild, y, data_ids, syn_ok = _wild_data(cfg)
    gen = np.array([r["generator"] for r in syn_rows])
    ys = np.array([r["label"] for r in syn_rows])
    gens = sorted(set(gen) - {"random_dir"})
    split = np.array([r["split"] for r in nat_rows])
    ev_idx = np.flatnonzero(wild & held)
    report = {"n_eval": int(len(ev_idx)), "eval_pos": int(y[ev_idx].sum()), "eval_pos_rate": float(y[ev_idx].mean()), "features": {}}
    for key in ("L15_cot_first32", "L15_cot_mean", "L15_final_mean"):
        Xn, Xs = nat_f[key].astype(np.float32), syn_f[key].astype(np.float32)
        ok_n, ok_s = ~np.isnan(Xn).any(1), ~np.isnan(Xs).any(1)
        ev = ev_idx[ok_n[ev_idx]]
        any_tr = np.flatnonzero((split == "train") & ~held & ok_n)
        wild_tr = np.flatnonzero(wild & ~held & ok_n)
        neg = np.flatnonzero(np.isin(pid, list(data_ids)) & ~held & (y == 0) & ok_n)
        pos = {g: np.flatnonzero((gen == g) & (ys == 1) & syn_ok & ok_s) for g in gens}

        def balanced(pool_idx):
            p, n = pool_idx[y[pool_idx] == 1], pool_idx[y[pool_idx] == 0]

            def fn(b, seed):
                rng = np.random.default_rng(seed)
                if min(len(p), len(n)) < b // 2:
                    return None
                i = np.r_[rng.choice(p, b // 2, replace=False), rng.choice(n, b // 2, replace=False)]
                return Xn[i], y[i]

            return fn

        def synthetic(g, nat_frac):
            p, n = pos[g], neg
            wp, wn = wild_tr[y[wild_tr] == 1], wild_tr[y[wild_tr] == 0]

            def fn(b, seed):
                rng = np.random.default_rng(100 + seed)
                k = int(b * nat_frac / 2)
                if len(p) < b // 2 - k or len(n) < b // 2 - k or min(len(wp), len(wn)) < k:
                    return None
                i_s, i_n = rng.choice(p, b // 2 - k, replace=False), rng.choice(n, b // 2 - k, replace=False)
                X, yy = [Xs[i_s], Xn[i_n]], [np.ones(len(i_s)), np.zeros(len(i_n))]
                if k:
                    j = np.r_[rng.choice(wp, k, replace=False), rng.choice(wn, k, replace=False)]
                    X.append(Xn[j])
                    yy.append(y[j])
                return np.concatenate(X), np.concatenate(yy).astype(int)

            return fn

        arms = {"natural_any": balanced(any_tr), "natural_wild": balanced(wild_tr)}
        for g in gens:
            arms[f"synthetic_{g}"] = synthetic(g, 0.0)
            arms[f"mix_{g}"] = synthetic(g, 0.5)
        report["features"][key] = run_arm_grid(arms, Xn[ev], y[ev], pid[ev], {"wildchat_heldout": np.ones(len(ev), bool)}, cfg.wild_budgets, cfg.seeds, n_boot=cfg.n_boot, n_jobs=cfg.n_jobs, reference="natural_wild")
        log(f"{key}: wildchat probes done")
    json.dump(report, open(cfg.path("probes_wildchat.json"), "w"), indent=1)


# ------------------------------------------------------------------ GPU: reachability


def collect_tokens(model, tok, rows, specs, positions, max_tokens, rng, steered=True, max_len=2048, token_budget=12000):
    """Residual activations (at the highest steered layer) of the chosen tokens of `rows`, teacher-forced,
    with the specs' vectors added (steered=True) or not. positions "all": CoT tokens of the full transcript
    (every position steered); "prompt": user tokens of the prompt only (the steered tokens)."""
    import torch

    from natsteer.model import Steerer, capture_resid

    layer = max(s["layer"] for s in specs)
    seqs = []
    for r in rows:
        if positions == "prompt":
            ids, lab = render(tok, r["user"], effort=sv.EFFORT)
            sel = "user"
        else:
            ids, lab = render(tok, r["user"], r.get("thinking") or "", r.get("final") or "", effort=sv.EFFORT)
            sel = "cot"
        ids, lab = ids[:max_len], lab[:max_len]
        keep = np.flatnonzero(np.asarray(lab) == sel)
        if len(keep):
            seqs.append((ids, keep))
    seqs.sort(key=lambda s: len(s[0]))
    chunks, i = [], 0
    pad = tok.pad_token_id
    while i < len(seqs):
        j = i + 1
        while j < len(seqs) and (j - i + 1) * len(seqs[j][0]) <= token_budget:
            j += 1
        batch = seqs[i:j]
        width = max(len(s[0]) for s in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, (s, _) in enumerate(batch):
            ids[b, : len(s)] = torch.tensor(s)
            mask[b, : len(s)] = 1
        with ExitStack() as stack:
            if steered:
                for sp in specs:
                    stack.enter_context(Steerer(model, tok, sp["layer"], sp["vec"].to(model.device, torch.bfloat16)))
            cap = stack.enter_context(capture_resid(model, (layer,)))
            with torch.no_grad():
                model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False, logits_to_keep=1)
            h = cap[layer]
            for b, (_, keep) in enumerate(batch):
                chunks.append(h[b, torch.as_tensor(keep, device=h.device)].float())
        i = j
    x = torch.cat(chunks) if chunks else torch.zeros(0, model.config.hidden_size, device=model.device)
    if len(x) > max_tokens:
        x = x[torch.as_tensor(rng.choice(len(x), max_tokens, replace=False), device=x.device)]
    return x, layer


def recon_rel_mse(sae, x, chunk=2048):
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(x), chunk):
            c = x[s : s + chunk]
            xh = sae.decode(sae.encode(c))
            out.append(((c - xh) ** 2).sum(-1) / (c**2).sum(-1).clamp(min=1e-9))
    return torch.cat(out)


def nn_rel_dist(q, bank, chunk=512):
    """Distance to the nearest bank activation, relative to the query norm."""
    import torch

    b2 = (bank**2).sum(-1)
    out = []
    with torch.no_grad():
        for s in range(0, len(q), chunk):
            c = q[s : s + chunk]
            d2 = (c**2).sum(-1, keepdim=True) + b2[None, :] - 2 * c @ bank.T
            out.append(d2.min(1).values.clamp(min=0).sqrt() / c.norm(dim=-1).clamp(min=1e-9))
    return torch.cat(out)


def stage_reach(cfg, args):
    import torch

    from natsteer.model import load_model
    from natsteer.sae import load_andyrdt_sae

    att = torch.load(cfg.path("attribution_wide.pt"))
    screen = json.load(open(cfg.path("latents_screen_table.json")))
    vtable = json.load(open(cfg.path("latents_validated_table.json")))
    choices = json.load(open(cfg.path("generator_choices_v1.json")))
    sweep = json.load(open(cfg.path("sweep_table.json")))
    _, _, data = sv.split_prompts(cfg)
    vconds, _ = sv.validate_conditions(att, screen, cfg, min_gain=cfg.refusal_gain, cap=cfg.validate_cap)
    vt = json.load(open(cfg.path("latents_validated_table.json")))
    gens = sv.build_generators(cfg, "refusal", att, choices, vt, sweep, data, sv.PROMPTED_REFUSAL)
    byname = {c["name"]: c for c in vconds}
    for g in gens:
        if g.get("_from"):
            g["specs"] = byname[g["_from"]]["specs"]
    val_rows = read_jsonl(cfg.path("latents_validated.jsonl"))
    syn_rows = read_jsonl(cfg.path("synthetic.jsonl"))
    by_cond = {}
    for r in val_rows + syn_rows:
        by_cond.setdefault(r["condition"], []).append(r)
    conds = [{"name": c["name"], "specs": c["specs"], "family": c.get("family"), "source": "validate", "stats": vtable.get(c["name"])} for c in vconds] + [{"name": g["name"], "specs": g["specs"], "family": g["generator"], "source": "generate", "stats": None} for g in gens if g["specs"]]
    if cfg.smoke:
        conds = conds[:4] + [c for c in conds if c["source"] == "generate"][:3]
    nat = [r for r in sv.load_natural(cfg) if r.get("final") and r["split"] == "train"]
    rng = np.random.default_rng(0)
    model, tok = load_model("bf16")
    saes = {}
    banks = {}
    nat_pick = [nat[i] for i in rng.choice(len(nat), min(len(nat), 800), replace=False)]
    floor = {}
    table = []
    t0 = time.time()
    for k, c in enumerate(conds):
        rows = by_cond.get(c["name"], [])
        specs = [{"layer": s["layer"], "vec": s["vec"], "positions": s["positions"]} for s in c["specs"]]
        if not rows or not specs:
            continue
        pos = specs[0]["positions"]
        layer = max(s["layer"] for s in specs)
        if layer not in saes:
            saes[layer] = load_andyrdt_sae(layer, 64, device=model.device)
        if (layer, pos) not in banks:  # bank and natural floor for this layer and token class
            x, _ = collect_tokens(model, tok, nat_pick[: len(nat_pick) * 3 // 4], [{"layer": layer, "vec": None}], pos, cfg.reach_bank_tokens, rng, steered=False)
            held, _ = collect_tokens(model, tok, nat_pick[len(nat_pick) * 3 // 4 :], [{"layer": layer, "vec": None}], pos, cfg.reach_nn_queries, rng, steered=False)
            banks[(layer, pos)] = x
            floor[(layer, pos)] = {"rel_mse": float(recon_rel_mse(saes[layer], held).mean()), "rel_nn": float(nn_rel_dist(held, x).median())}
        pick = [rows[i] for i in rng.choice(len(rows), min(len(rows), cfg.reach_rows), replace=False)]
        xs, _ = collect_tokens(model, tok, pick, specs, pos, cfg.reach_nn_queries * 2, rng, steered=True)
        xu, _ = collect_tokens(model, tok, pick, specs, pos, cfg.reach_nn_queries * 2, rng, steered=False)
        n = min(len(xs), len(xu))
        bank = banks[(layer, pos)]
        e = {"name": c["name"], "family": c["family"], "source": c["source"], "layer": layer, "positions": pos, "n_specs": len(specs), "n_tokens": int(n), "vec_norm": float(sum(s["vec"].float().norm() for s in specs)), "stats": c["stats"],
             "rel_mse_steered": float(recon_rel_mse(saes[layer], xs).mean()), "rel_mse_unsteered": float(recon_rel_mse(saes[layer], xu).mean()),
             "rel_nn_steered": float(nn_rel_dist(xs[: cfg.reach_nn_queries], bank).median()), "rel_nn_unsteered": float(nn_rel_dist(xu[: cfg.reach_nn_queries], bank).median()),
             "act_norm": float(xu.norm(dim=-1).median()), "natural_floor": floor[(layer, pos)]}
        table.append(e)
        log(f"{k + 1}/{len(conds)} {c['name']}: recon {e['rel_mse_unsteered']:.3f}->{e['rel_mse_steered']:.3f} nn {e['rel_nn_unsteered']:.3f}->{e['rel_nn_steered']:.3f} ({time.time() - t0:.0f}s)")
    json.dump(table, open(cfg.path("reach_table.json"), "w"), indent=1)


# ------------------------------------------------------------------ GPU: steering timing with HF masks


def stage_timing_hf(cfg, args):
    import torch

    from natsteer.model import Steerer, generate, load_model

    att = torch.load(cfg.path("attribution_wide.pt"))
    screen = json.load(open(cfg.path("latents_screen_table.json")))
    vt = json.load(open(cfg.path("latents_validated_table.json")))
    choices = json.load(open(cfg.path("generator_choices_v1.json")))
    sweep = json.load(open(cfg.path("sweep_table.json")))
    D = torch.load(cfg.path("directions.pt"))["dirs"]
    _, val, data = sv.split_prompts(cfg)
    vconds, _ = sv.validate_conditions(att, screen, cfg, min_gain=cfg.refusal_gain, cap=cfg.validate_cap)
    byname = {c["name"]: c for c in vconds}
    base = vt["base"]
    ranked = sorted([c for c, s in vt.items() if s.get("family") in ("attr", "bundle") and sv.acceptable(s, base) and c in byname], key=lambda c: -vt[c]["pos_rate"])
    sets = [(c, byname[c]["specs"]) for c in ranked[: cfg.timing_sets - 1]]
    if "dom_all" in choices:
        s = sweep[choices["dom_all"]]
        sets.append((choices["dom_all"], [{"layer": s["layer"], "vec": s["coeff"] * D[s["kind"]][s["layer"]], "positions": "all"}]))
    prompts_rows = val[: cfg.timing_prompts]
    model, tok = load_model("bf16")
    prompts = [render(tok, p["user"], effort=sv.EFFORT)[0] for p in prompts_rows]
    path = cfg.path("timing_hf.jsonl")
    rows_all = read_jsonl(path)
    done = {r["condition"] for r in rows_all}
    for name, specs in sets:
        for seg, kw in (("prompt", {"prefill_only": True}), ("cot", {"segments": ("cot",)}), ("final", {"segments": ("final",)})):
            cname = f"hf_{name}_{seg}"
            if cname in done:
                continue
            t0 = time.time()
            with ExitStack() as stack:
                st = [stack.enter_context(Steerer(model, tok, sp["layer"], sp["vec"].to(model.device, torch.bfloat16), **kw)) for sp in specs]
                gens = generate(model, tok, prompts, max_new_tokens=cfg.timing_tokens, temperature=1.0, steerer=st[0], batch_size=32, seed=0)
            rows = [{"id": f"{p['prompt_id']}|{cname}", "prompt_id": p["prompt_id"], "user": p["user"], "effort": sv.EFFORT, "condition": cname, "family": "timing", "kind": name, "positions": seg} | sv.refusal_labeler(tok, p, g) for p, g in zip(prompts_rows, gens)]
            rows_all += rows
            write_jsonl(path, rows_all)
            s = sv.summarize(rows)
            log(f"{cname}: pos {s['pos_rate']:.2f} incomplete {s['incomplete_rate']:.2f} rep4 {s['rep4']:.3f} ({time.time() - t0:.0f}s)")
    json.dump(sv.condition_table(rows_all), open(cfg.path("timing_table.json"), "w"), indent=1)


def stage_summary(cfg, args):
    import subprocess

    subprocess.run([sys.executable, str(Path(__file__).parent / "morning_summary.py"), str(cfg.dir)], check=True)


def stage_report_extra(cfg, args):
    out = run_dir("steer_v1_extra_smoke" if cfg.smoke else "steer_v1_extra")
    for name in ("reach_table.json", "timing_table.json", "laser_warmstart.json", "probes_wildchat.json", "SUMMARY.md"):
        if cfg.path(name).exists():
            shutil.copy(cfg.path(name), out / name)
    (out / "report.done").write_text("ok")
    log("saved", out)


STAGES = {
    "laser_warmstart": (stage_laser_warmstart, "laser_warmstart.json"),
    "probes_wildchat": (stage_probes_wildchat, "probes_wildchat.json"),
    "reach": (stage_reach, "reach_table.json"),
    "timing_hf": (stage_timing_hf, "timing_table.json"),
    "summary": (stage_summary, "summary.never"),
    "report_extra": (stage_report_extra, "report_extra.never"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = XCfg(args.smoke)
    fn, marker = STAGES[args.stage]
    if cfg.path(marker).exists() and not args.force:
        log(f"{args.stage}: {marker} exists, skipping")
        return
    t0 = time.time()
    fn(cfg, args)
    log(f"stage {args.stage} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
