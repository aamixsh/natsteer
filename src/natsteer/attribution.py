"""SAE latent attribution (OpenAI, "Debugging misaligned completions with sparse-autoencoder
latent attribution", Dec 2025), contrastive Delta-attribution over paired completions.

For a completion with log-loss L and resid_post a_t at a layer, mean-ablating latent i's
decoder direction d_i (unit norm) changes the loss by about
    delta_{i,t} = -(g_t . d_i) * ((a_t - abar) . d_i),   g_t = dL/da_t,
so delta > 0 means the latent supports the completion. Per completion we average over
completion tokens. Expanding the product, mean_t delta = -S1 + (abar . d_i) * S2 with
S1 = mean_t (g.d)(a.d) and S2 = mean_t (g.d), so abar can be fixed after the pass.
Delta-attribution of a behavior = mean over pairs of [delta(behavior completion) -
delta(contrast completion of the same prompt)] = mean over pairs of (-A + abar * B) with
A = S1(pos) - S1(neg), B = S2(pos) - S2(neg). Only running sums of A, B, A^2, B^2, AB are kept
(five vectors per layer), so memory does not grow with the number of pairs and any set of
layers can be run in one pass.
"""

import numpy as np
import torch

from natsteer.harmony import render
from natsteer.model import get_layer


def completion_start(labels) -> int:
    """First generated position: right after `<|start|>assistant` that follows the user turn."""
    lab = np.asarray(labels)
    return int(np.flatnonzero(lab == "user")[-1]) + 4


def per_sample_stats(model, tok, row, saes: dict, max_len: int = 2048, chunk: int = 256) -> dict:
    """One forward and backward pass; returns per-layer S1, S2 [d_sae], abar sums, SAE code stats."""
    ids, lab = render(tok, row["user"], row.get("thinking"), row.get("final"), effort=row.get("effort", "medium"))
    ids, lab = ids[:max_len], lab[:max_len]
    start = completion_start(lab)
    if start >= len(ids) - 1:
        return None
    layers = sorted(saes)
    acts = {}

    def make_hook(L, leaf):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if leaf:
                h = h.detach().requires_grad_(True)
            else:
                h.retain_grad()
            acts[L] = h
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        return hook

    handles = [get_layer(model, L).register_forward_hook(make_hook(L, i == 0)) for i, L in enumerate(layers)]
    try:
        x = torch.tensor([ids], device=model.device)
        with torch.enable_grad():
            logits = model(input_ids=x).logits[0]
            tgt = x[0, start:]
            loss = torch.nn.functional.cross_entropy(logits[start - 1 : -1].float(), tgt)
            loss.backward()
    finally:
        for h in handles:
            h.remove()
    out = {"loss": float(loss), "n_tokens": len(ids) - start}
    for L in layers:
        sae = saes[L]
        a = acts[L][0, start:].detach().float()
        g = acts[L].grad[0, start:].float()
        D = sae.W_dec.float()
        Dn = D / D.norm(dim=1, keepdim=True)
        S1 = torch.zeros(Dn.shape[0], device=a.device)
        S2 = torch.zeros_like(S1)
        code_sum = torch.zeros_like(S1)
        code_cnt = torch.zeros_like(S1)
        for s in range(0, a.shape[0], chunk):
            ga, aa = g[s : s + chunk] @ Dn.T, a[s : s + chunk] @ Dn.T
            S1 += (ga * aa).sum(0)
            S2 += ga.sum(0)
            f = sae.encode(a[s : s + chunk])
            code_sum += f.sum(0)
            code_cnt += (f > 0).sum(0)
        T = a.shape[0]
        out[L] = {"S1": (S1 / T).cpu(), "S2": (S2 / T).cpu(), "a_sum": a.sum(0).cpu(), "code_mean": (code_sum / T).cpu(), "code_sum": code_sum.cpu(), "code_cnt": code_cnt.cpu()}
    model.zero_grad(set_to_none=True)
    return out


def delta_attribution(model, tok, pairs, saes: dict, log_every: int = 50) -> dict:
    """pairs: list of (behavior_row, contrast_row) on the same prompt. Returns per-layer results."""
    import time

    layers = sorted(saes)
    z = lambda L: torch.zeros(saes[L].W_dec.shape[0], dtype=torch.float64)  # noqa: E731
    acc = {L: {k: z(L) for k in ("A", "B", "AA", "BB", "AB", "dact", "code_sum", "code_cnt")} | {"a_sum": 0, "n_tok": 0} for L in layers}
    t0 = time.time()
    kept = 0
    for k, (pos, neg) in enumerate(pairs):
        sp, sn = per_sample_stats(model, tok, pos, saes), per_sample_stats(model, tok, neg, saes)
        if sp is None or sn is None:
            continue
        kept += 1
        for L in layers:
            A = acc[L]
            a_, b_ = (sp[L]["S1"] - sn[L]["S1"]).double(), (sp[L]["S2"] - sn[L]["S2"]).double()
            A["A"] += a_
            A["B"] += b_
            A["AA"] += a_ * a_
            A["BB"] += b_ * b_
            A["AB"] += a_ * b_
            A["dact"] += (sp[L]["code_mean"] - sn[L]["code_mean"]).double()
            for s in (sp, sn):
                A["a_sum"] = A["a_sum"] + s[L]["a_sum"].double()
                A["n_tok"] += s["n_tokens"]
                A["code_sum"] += s[L]["code_sum"].double()
                A["code_cnt"] += s[L]["code_cnt"].double()
        if log_every and (k + 1) % log_every == 0:
            print(f"attribution: {k + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    n = kept
    res = {"n_pairs": kept}
    for L in layers:
        A = acc[L]
        abar = (A["a_sum"] / A["n_tok"]).float()
        D = saes[L].W_dec.float().cpu()
        abar_proj = ((D / D.norm(dim=1, keepdim=True)) @ abar).double()
        mean = (-A["A"] + abar_proj * A["B"]) / n
        # variance of diff_i = -A_i + c B_i  =  var(A) - 2c cov(A,B) + c^2 var(B)   (sample variances)
        var = lambda S, SS: (SS - S * S / n) / max(1, n - 1)  # noqa: E731
        cov = (A["AB"] - A["A"] * A["B"] / n) / max(1, n - 1)
        v = var(A["A"], A["AA"]) - 2 * abar_proj * cov + abar_proj**2 * var(A["B"], A["BB"])
        res[L] = {
            "delta_attr": mean.float(),
            "delta_attr_t": (mean / (v.clamp(min=0).sqrt() / n**0.5 + 1e-12)).float(),
            "delta_act": (A["dact"] / n).float(),
            "act_mean_active": (A["code_sum"] / A["code_cnt"].clamp(min=1)).float(),
            "act_freq": (A["code_cnt"] / A["n_tok"]).float(),
            "abar": abar,
        }
    return res
