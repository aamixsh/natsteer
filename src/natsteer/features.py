"""Batched pooled-feature extraction from clean (unsteered) forward passes.

Rows are dicts with `user` and optionally `thinking`, `final`, `effort`. Each row is
rendered alone, batches are right-padded (real tokens never attend to pads), and
pooled features are stored as float16. Batched and single-row features agree to
within the recomputation noise floor (cosine above 0.99).
"""

import time

import numpy as np
import torch

from natsteer.harmony import render
from natsteer.model import capture_resid
from natsteer.probes import pool as pool_acts


@torch.no_grad()
def extract(model, tok, rows, layers, poolings=("user_mean", "prompt_last"), token_budget=16384, max_len=4096, log_every=2000):
    rendered = []
    for r in rows:
        ids, lab = render(tok, r["user"], r.get("thinking"), r.get("final"), effort=r.get("effort", "medium"))
        rendered.append((ids[:max_len], lab[:max_len]))
    order = np.argsort([len(ids) for ids, _ in rendered])
    out = {L: {p: np.full((len(rows), model.config.hidden_size), np.nan, np.float16) for p in poolings} for L in layers}
    i, t0, done = 0, time.time(), 0
    with capture_resid(model, layers) as cap:
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(rendered[order[j]][0]) <= token_budget:
                j += 1
            batch = order[i:j]
            width = max(len(rendered[k][0]) for k in batch)
            ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for b, k in enumerate(batch):
                n = len(rendered[k][0])
                ids[b, :n] = torch.tensor(rendered[k][0])
                mask[b, :n] = 1
            model.model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False)
            for L in layers:
                for b, k in enumerate(batch):
                    n = len(rendered[k][0])
                    for p in poolings:
                        out[L][p][k] = pool_acts(cap[L][b, :n], rendered[k][1], p).cpu().numpy()
            done += len(batch)
            i = j
            if log_every and done % log_every < len(batch):
                print(f"{done}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return out


def save(path, rows, feats):
    np.savez(path, ids=np.array([r["id"] for r in rows]), **{f"L{L}_{p}": v for L, d in feats.items() for p, v in d.items()})
