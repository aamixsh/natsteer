"""E0 smoke test: model + hooks + SAEs + probes + generation + steering on one GPU.

Writes report.json to data/runs/e0_smoke/<UTC stamp>/.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/e0_smoke.py [--n-convos 100]
"""

import argparse
import gc
import json
import platform
import random
import time
from collections import Counter

import numpy as np
import torch

from natsteer.config import run_dir, scratch_dir
from natsteer.data import load_xstest
from natsteer.harmony import SegmentTracker, render, split_generation
from natsteer.model import Steerer, _hidden, capture_resid, load_model
from natsteer.probes import auroc, oof_scores, pool, tpr_at_fpr
from natsteer.sae import load_andyrdt_sae, load_goodfire_sae, recon_stats

LAYERS = (15, 19)
SEGMENTS = ("system", "user", "cot", "final", "scaffold")


def gb(x):
    return round(x / 2**30, 2)


def gpu_mem():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_gb": gb(torch.cuda.memory_allocated()), "peak_gb": gb(torch.cuda.max_memory_allocated()), "device_free_gb": gb(free), "device_total_gb": gb(total)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def collect(model, convos, max_len):
    """Teacher-forced pass over full conversations; returns per-layer CPU activations and labels."""
    acts = {L: [] for L in LAYERS}
    labels, nll = [], {"cot": [0.0, 0], "final": [0.0, 0]}
    with capture_resid(model, LAYERS) as cap:
        for r in convos:
            ids, lab = render(tok, r["user"], r["thinking"], r["final"], effort=r["effort"])
            ids, lab = ids[:max_len], lab[:max_len]
            x = torch.tensor([ids], device=model.device)
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
            # Next-token loss on the model's own CoT and answer: a correctness check for the dequantized weights.
            for seg in nll:
                pos = [i for i in range(len(ids) - 1) if lab[i + 1] == seg]
                if pos:
                    logits = model.lm_head(h[pos]).float()
                    nll[seg][0] += torch.nn.functional.cross_entropy(logits, x[0, [p + 1 for p in pos]], reduction="sum").item()
                    nll[seg][1] += len(pos)
            for L in LAYERS:
                acts[L].append(cap[L][0].to("cpu"))
            labels.append(lab)
    return acts, labels, {k: v[0] / max(v[1], 1) for k, v in nll.items()}


def by_segment(acts_list, labels, seg):
    rows = [a[1:][np.asarray(l[1:]) == seg] for a, l in zip(acts_list, labels)]  # drop position 0
    return torch.cat(rows) if rows else torch.empty(0)


def sae_report(sae, acts_list, labels):
    out = {seg: recon_stats(sae, by_segment(acts_list, labels, seg)) for seg in SEGMENTS}
    out["all_but_system"] = recon_stats(sae, torch.cat([by_segment(acts_list, labels, s) for s in ("user", "cot", "final", "scaffold")]))
    return out


@torch.no_grad()
def batch_independence(sae, x):
    """Codes for a token must not depend on what else is in the batch (fixed threshold, not batch top-k)."""
    x = x.to(next(sae.parameters()).device, torch.float32)
    big = sae.encode(x)
    small = torch.cat([sae.encode(x[i : i + 7]) for i in range(0, len(x), 7)])
    support_agree = ((big > 0) == (small > 0)).float().mean().item()
    batch_topk = sae.encode(x, use_threshold=False)
    batch_topk_small = torch.cat([sae.encode(x[i : i + 7], use_threshold=False) for i in range(0, len(x), 7)])
    return {
        "threshold": float(sae.threshold),
        "threshold_max_abs_diff": (big - small).abs().max().item(),
        "threshold_support_agreement": support_agree,
        "batch_topk_support_agreement": ((batch_topk > 0) == (batch_topk_small > 0)).float().mean().item(),
        "l0_threshold": (big > 0).float().sum(-1).mean().item(),
        "l0_batch_topk": (batch_topk > 0).float().sum(-1).mean().item(),
    }


@torch.no_grad()
def prompt_probes(model, rows):
    feats = {(L, p): [] for L in LAYERS for p in ("user_mean", "prompt_last")}
    with capture_resid(model, LAYERS) as cap:
        for r in rows:
            ids, lab = render(tok, r["user"], effort=r["effort"])
            model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            for (L, p) in feats:
                feats[(L, p)].append(pool(cap[L][0], lab, p).cpu())
    y = np.array([r["label"] for r in rows])
    out = {}
    for (L, p), v in feats.items():
        s = oof_scores(torch.stack(v).numpy(), y)
        out[f"L{L}/{p}"] = {"auroc": auroc(y, s), "tpr@0.05": tpr_at_fpr(y, s, 0.05), "tpr@0.01": tpr_at_fpr(y, s, 0.01)}
    return out


@torch.no_grad()
def generate(model, rows, max_new_tokens, steerer=None):
    prompts = [render(tok, r["user"], effort="low")[0] for r in rows]
    width = max(map(len, prompts))
    ids = torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in prompts], device=model.device)
    mask = (ids != tok.pad_token_id).long()
    t0 = time.time()
    if steerer is not None:
        steerer.reset()
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    gens = [split_generation(tok, row[width:].tolist()) for row in out]
    n_new = int((out[:, width:] != tok.pad_token_id).sum())
    seg_counts = Counter()
    for row in out.tolist():
        seg_counts.update(SegmentTracker(tok, pad_id=tok.pad_token_id).feed_many(row)[width:])
    return gens, {"seconds": round(dt, 1), "new_tokens": n_new, "tok_per_s": round(n_new / dt, 1), "generated_by_segment": dict(seg_counts)}


@torch.no_grad()
def steering_checks(model, convo):
    ids, lab = render(tok, convo["user"], convo["thinking"], convo["final"], effort=convo["effort"])
    ids, lab = ids[:768], lab[:768]
    x = torch.tensor([ids], device=model.device)
    lab_arr = np.asarray(lab)
    first_cot = int(np.flatnonzero(lab_arr == "cot")[0])
    with capture_resid(model, (15, 19)) as cap:
        base_logits = model(input_ids=x, use_cache=False).logits.float()
        base19 = cap[19].clone()
        norm15 = cap[15][0, 1:].float().norm(dim=-1).median().item()
        zero = torch.zeros(model.config.hidden_size, device=model.device)
        with Steerer(model, tok, 15, zero):
            zero_logits = model(input_ids=x, use_cache=False).logits.float()
        g = torch.Generator().manual_seed(0)
        direction = torch.randn(model.config.hidden_size, generator=g)
        vec = (direction / direction.norm() * 0.5 * norm15).to(model.device)
        with Steerer(model, tok, 15, vec, segments={"cot"}) as st:
            model(input_ids=x, use_cache=False)
            counts = dict(st.steered)
        st19 = cap[19].clone()
        model(input_ids=x[:, :first_cot], use_cache=False)
        prefix19 = cap[19]
    # Noise floor: the same prefix computed alone vs inside the full sequence.
    floor = ((prefix19[0, 1:] - base19[0, 1:first_cot]).float().norm(dim=-1) / base19[0, 1:first_cot].float().norm(dim=-1)).cpu().numpy()
    # Relative change per position at layer 19. Unsteered positions are not exactly
    # unchanged: the bf16 MoE runs one matmul per expert, sized by how many tokens
    # routed to it, so rerouting the steered tokens changes rounding elsewhere.
    rel = ((st19[0] - base19[0]).float().norm(dim=-1) / base19[0].float().norm(dim=-1)).cpu().numpy()
    pre = rel[1:first_cot]
    return {
        "median_norm_L15": norm15,
        "zero_vector_max_logit_diff": (zero_logits - base_logits).abs().max().item(),
        "pre_cot_rel_diff_L19": {"median": float(np.median(pre)), "p99": float(np.quantile(pre, 0.99)), "max": float(pre.max())},
        "noise_floor_prefix_alone_rel_diff_L19": {"median": float(np.median(floor)), "p99": float(np.quantile(floor, 0.99)), "max": float(floor.max())},
        "cot_rel_diff_L19": {"median": float(np.median(rel[lab_arr == "cot"])), "min": float(rel[lab_arr == "cot"].min())},
        "cot_only_steered_counts": counts,
        "n_cot_tokens": int((lab_arr == "cot").sum()),
    }


def attribution_check(model, sae, convo, acts15):
    """Backprop CoT next-token loss to layer-15 resid_post and score SAE latents.

    delta_{i,t} = -(g_t . d_i)((a_t - a_bar) . d_i), averaged over tokens, as in the
    OpenAI SAE latent attribution post. Also checks causality of the gradient:
    positions after the last scored prediction must get exactly zero gradient.
    """
    ids, lab = render(tok, convo["user"], convo["thinking"], convo["final"], effort=convo["effort"])
    ids, lab = ids[:1024], lab[:1024]
    x = torch.tensor([ids], device=model.device)
    store = {}

    def leaf(_m, _i, out):
        h = _hidden(out).detach().requires_grad_(True)
        store["a"] = h
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    pred_pos = [i for i in range(len(ids) - 1) if lab[i + 1] == "cot"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    handle = model.model.layers[15].register_forward_hook(leaf)
    try:
        with torch.enable_grad():
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
            logits = model.lm_head(h[pred_pos]).float()
            loss = torch.nn.functional.cross_entropy(logits, x[0, [p + 1 for p in pred_pos]])
            loss.backward()
    finally:
        handle.remove()
    g = store["a"].grad[0].float()  # [seq, d_model]
    a = store["a"][0].detach().float()
    a_bar = torch.cat([t[1:] for t in acts15]).float().mean(0).to(a.device)
    W = sae.W_dec.float()  # [d_sae, d_model]
    tok_idx = torch.arange(1, max(pred_pos) + 1, device=a.device)
    gd, ad = g[tok_idx] @ W.T, (a[tok_idx] - a_bar) @ W.T
    score = -(gd * ad).mean(0)
    top = score.topk(10)
    after = g[max(pred_pos) + 1 :]
    return {
        "loss": loss.item(),
        "n_scored_tokens": len(pred_pos),
        "seconds": round(time.time() - t0, 2),
        "peak_gb": gb(torch.cuda.max_memory_allocated()),
        "grad_finite": bool(torch.isfinite(g).all()),
        "grad_norm_median": g[tok_idx].norm(dim=-1).median().item(),
        "grad_after_last_prediction_max_abs": after.abs().max().item() if len(after) else None,
        "top10_latents": top.indices.tolist(),
        "top10_scores": [round(v, 5) for v in top.values.tolist()],
    }


def model_stage(args, report, out_dir, scratch):
    """Everything that needs the LM. Returns activations; the model is freed on return."""
    global tok
    model, tok = load_model(args.mode)
    report["model"] = {"mode": args.mode, "load_seconds": round(model.load_seconds, 1), "attn": model.config._attn_implementation} | gpu_mem()
    log(f"model loaded in {model.load_seconds:.0f}s: {report['model']}")

    rows = load_xstest()
    rng = random.Random(0)
    convos = rng.sample([r for r in rows if r["label"] == 0], args.n_convos // 2) + rng.sample([r for r in rows if r["label"] == 1], args.n_convos // 2)

    t0 = time.time()
    acts, labels, nll = collect(model, convos, args.max_len)
    seg_counts = {s: int(sum((np.asarray(l) == s).sum() for l in labels)) for s in SEGMENTS}
    report["collect"] = {"seconds": round(time.time() - t0, 1), "tokens_by_segment": seg_counts, "nll_per_token": nll} | gpu_mem()
    log(f"collected activations: {report['collect']}")
    torch.save({"acts": acts, "labels": labels}, scratch / "acts.pt")

    t0 = time.time()
    report["probes_xstest_cv"] = prompt_probes(model, rows)
    log(f"prompt probes ({time.time() - t0:.0f}s): {json.dumps(report['probes_xstest_cv'])}")

    report["steering"] = steering_checks(model, convos[0])
    log(f"steering checks: {report['steering']}")

    gen_rows = convos[:2] + convos[-2:]
    gens, speed = generate(model, gen_rows, args.max_new_tokens)
    report["generation"] = {"speed": speed, "has_cot": [bool(g.get("cot")) for g in gens], "has_final": [bool(g.get("final")) for g in gens]} | gpu_mem()
    log(f"generation: {report['generation']}")
    vec = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    vec = (vec / vec.norm() * 0.5 * report["steering"]["median_norm_L15"]).to(model.device)
    with Steerer(model, tok, 15, vec, segments={"cot"}) as st:
        _, speed_st = generate(model, gen_rows, 64, steerer=st)
        report["generation_steered_cot_only"] = {"speed": speed_st, "steered_counts": dict(st.steered)}
    log(f"steered generation: {report['generation_steered_cot_only']}")
    with open(out_dir / "generations.json", "w") as f:
        json.dump([{"user": r["user"], "label": r["label"], **g} for r, g in zip(gen_rows, gens)], f, indent=1)

    if args.keep_model:
        # Model and an SAE together, as attribution and GCG will need. Needs a mostly free GPU.
        sae = load_andyrdt_sae(15, 64)
        report["model_plus_sae"] = gpu_mem()
        log(f"model and L15 SAE loaded together: {report['model_plus_sae']}")
        report["attribution_check"] = attribution_check(model, sae, convos[-1], acts[15])
        log(f"attribution check: {report['attribution_check']}")
        del sae
    return acts, labels


def sae_stage(acts, labels, report):
    report["sae"] = {}
    for L in LAYERS:
        for k in (64, 128):
            t0 = time.time()
            sae = load_andyrdt_sae(L, k)
            name = f"andyrdt_L{L}_k{k}"
            stats = sae_report(sae, acts[L], labels)
            stats["batch_independence"] = batch_independence(sae, by_segment(acts[L], labels, "cot")[:1024])
            stats["load_and_eval_seconds"] = round(time.time() - t0, 1)
            stats["gpu"] = gpu_mem()
            report["sae"][name] = stats
            log(f"{name}: FVE by segment " + ", ".join(f"{s}={stats[s].get('frac_variance_explained', float('nan')):.3f}" for s in SEGMENTS) + f"; batch check {stats['batch_independence']}")
            del sae
            gc.collect(); torch.cuda.empty_cache()
    for sub in (True, False):
        sae = load_goodfire_sae(subtract_b_dec=sub)
        name = f"goodfire_L15_k32_subtract_bdec={sub}"
        report["sae"][name] = sae_report(sae, acts[15], labels)
        log(f"{name}: FVE " + ", ".join(f"{s}={report['sae'][name][s].get('frac_variance_explained', float('nan')):.3f}" for s in SEGMENTS))
        del sae
        gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convos", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--mode", default="bf16", choices=["bf16", "mxfp4"])
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--keep-model", action="store_true", help="also load an SAE next to the model (needs about 45 GB free)")
    ap.add_argument("--acts", help="skip the model stage and evaluate SAEs on a saved acts.pt")
    args = ap.parse_args()

    out_dir = run_dir("e0_smoke")
    report = {"args": vars(args), "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    report["gpu_before"] = gpu_mem()
    if args.acts:
        saved = torch.load(args.acts, weights_only=False)
        acts, labels = saved["acts"], saved["labels"]
        log(f"run dir {out_dir}; SAE stage only, activations from {args.acts}")
    else:
        scratch = scratch_dir("e0smoke-acts")
        log(f"run dir {out_dir}; scratch {scratch}; GPU free {report['gpu_before']['device_free_gb']} GB")
        acts, labels = model_stage(args, report, out_dir, scratch)
        gc.collect(); torch.cuda.empty_cache()
        log(f"model freed: {gpu_mem()}")
    sae_stage(acts, labels, report)

    report["gpu_end"] = gpu_mem()
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    log(f"done; report at {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
