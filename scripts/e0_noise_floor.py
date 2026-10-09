"""Recomputation noise floor: the same prefix run alone vs inside a longer sequence.

Mathematically identical under causal attention, so any difference is numerics.
Reports per-layer relative difference and how often MoE routing (top-4 set) flips.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/e0_noise_floor.py
"""

import json
import random

import numpy as np
import torch

from natsteer.config import N_LAYERS, run_dir
from natsteer.data import load_xstest
from natsteer.harmony import render
from natsteer.model import capture_resid, load_model


@torch.no_grad()
def run(model, ids):
    routes = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def hook(_m, _i, out, i=i):
            routes[i] = out[2].sort(-1).values.cpu()  # router indices, [tokens, top_k]
        handles.append(layer.mlp.router.register_forward_hook(hook))
    with capture_resid(model, range(N_LAYERS)) as cap:
        model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
        acts = {L: cap[L][0].float().cpu() for L in cap}
    for h in handles:
        h.remove()
    return acts, routes


def main():
    model, tok = load_model("bf16")
    rows = load_xstest()
    random.Random(0).shuffle(rows)
    per_layer = {L: [] for L in range(N_LAYERS)}
    flips = {L: [] for L in range(N_LAYERS)}
    for r in rows[:5]:
        ids, lab = render(tok, r["user"], r["thinking"], r["final"], effort=r["effort"])
        ids = ids[:1024]
        cut = int(np.flatnonzero(np.asarray(lab) == "cot")[0])
        full, rf = run(model, ids)
        pre, rp = run(model, ids[:cut])
        for L in range(N_LAYERS):
            a, b = pre[L][1:cut], full[L][1:cut]
            per_layer[L] += ((a - b).norm(dim=-1) / b.norm(dim=-1)).tolist()
            flips[L] += (rp[L][1:cut] != rf[L][1:cut]).any(-1).tolist()
    report = {
        L: {
            "rel_diff_median": float(np.median(per_layer[L])),
            "rel_diff_p99": float(np.quantile(per_layer[L], 0.99)),
            "rel_diff_max": float(np.max(per_layer[L])),
            "routing_flip_rate": float(np.mean(flips[L])),
        }
        for L in range(N_LAYERS)
    }
    for L, v in report.items():
        print(L, {k: round(x, 4) for k, x in v.items()}, flush=True)
    out = run_dir("e0_noise_floor")
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print("saved", out / "report.json")


if __name__ == "__main__":
    main()
