"""Clean re-read of conversations: residual activations of the unmodified model, pooled into windows.

Each conversation is rendered by convo.render_with_turns and run through the model without any hooks other
than the residual capture. The assistant tokens of every turn are taken in order and cut at every segment
change (reasoning "cot" vs reply "final"); each run is split into windows of W consecutive tokens (the last
window of a run may be shorter). A window's feature is the mean resid_post over its tokens; a turn's feature is
the mean over all its assistant tokens. A monitor scores windows; a turn's score is its maximum window score.

Sequences longer than `chunk` tokens run alone, chunk by chunk with a KV cache, so attention memory stays
bounded (eager attention; gpt-oss has no SDPA path in transformers). Rendered sequences are cut at `max_len`;
windows after the cut are lost and the conversation is marked truncated.

Saved arrays (npz): ids [n_conv]; n_tok, truncated [n_conv]; w_conv, w_turn, w_seg (0 cot, 1 final), w_pos
(window index within the turn), w_len [n_win]; W{L} [n_win, d] float16; t_conv, t_turn [n_turn];
T{L} [n_turn, d] float16; norm{L} [n_conv] median token norm over assistant tokens.
"""

import time

import numpy as np
import torch

from natsteer.convo import render_with_turns
from natsteer.model import capture_resid

SEG_CODE = {"cot": 0, "final": 1}


def windows_of(seg, turn, W: int = 32):
    """[(turn, seg_code, pos_in_turn, token index array)] over assistant tokens, in order."""
    seg = np.asarray(seg)
    turn = np.asarray(turn)
    asst = np.flatnonzero(np.isin(seg, ("cot", "final")) & (turn >= 0))
    out, counters = [], {}
    if len(asst) == 0:
        return out
    # runs of consecutive positions with the same (turn, segment)
    keys = [(int(turn[i]), seg[i]) for i in asst]
    start = 0
    for j in range(1, len(asst) + 1):
        if j == len(asst) or keys[j] != keys[start] or asst[j] != asst[j - 1] + 1:
            t, s = keys[start]
            run = asst[start:j]
            for a in range(0, len(run), W):
                k = counters.get(t, 0)
                out.append((t, SEG_CODE[s], k, run[a : a + W]))
                counters[t] = k + 1
            start = j
    return out


def _pool(acts, rendered, W):
    """acts: {L: [n, d] tensor on GPU} for one conversation -> window and turn features (numpy float16)."""
    ids, seg, turn = rendered
    wins = windows_of(seg, turn, W)
    layers = sorted(acts)
    win_feats = {L: np.zeros((len(wins), acts[L].shape[-1]), np.float16) for L in layers}
    for L in layers:
        a = acts[L].float()
        for i, (_, _, _, idx) in enumerate(wins):
            win_feats[L][i] = a[torch.as_tensor(idx, device=a.device)].mean(0).cpu().numpy()
    turns = sorted({t for t, *_ in wins})
    turn_feats = {L: np.zeros((len(turns), acts[L].shape[-1]), np.float16) for L in layers}
    seg_a, turn_a = np.asarray(seg), np.asarray(turn)
    for L in layers:
        a = acts[L].float()
        for i, t in enumerate(turns):
            idx = np.flatnonzero((turn_a == t) & np.isin(seg_a, ("cot", "final")))
            turn_feats[L][i] = a[torch.as_tensor(idx, device=a.device)].mean(0).cpu().numpy()
    meta = np.array([(t, s, k, len(idx)) for t, s, k, idx in wins], dtype=np.int32).reshape(-1, 4)
    a_idx = torch.as_tensor(np.flatnonzero(np.isin(seg_a, ("cot", "final")) & (turn_a >= 0)), device=acts[layers[0]].device)
    norms = {L: float(acts[L][a_idx].float().norm(dim=-1).median()) if len(a_idx) else float("nan") for L in layers}
    return meta, win_feats, np.array(turns, dtype=np.int32), turn_feats, norms


@torch.no_grad()
def _forward_chunked(model, ids, layers, chunk):
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    parts = {L: [] for L in layers}
    x = torch.tensor(ids, device=model.device)[None]
    with capture_resid(model, layers) as cap:
        for s in range(0, x.shape[1], chunk):
            model.model(input_ids=x[:, s : s + chunk], past_key_values=cache, use_cache=True)
            for L in layers:
                parts[L].append(cap[L][0].clone())
    return {L: torch.cat(parts[L]) for L in layers}


@torch.no_grad()
def extract(model, tok, convs, layers=(11, 15, 19), W: int = 32, token_budget: int = 16384, chunk: int = 4096, max_len: int = 32768, log_every: int = 200):
    rendered = []
    n_tok, truncated = [], []
    for c in convs:
        ids, seg, turn = render_with_turns(tok, c)
        truncated.append(len(ids) > max_len)
        rendered.append((ids[:max_len], seg[:max_len], turn[:max_len]))
        n_tok.append(min(len(ids), max_len))
    order = np.argsort(n_tok)
    results = [None] * len(convs)
    t0, done, i = time.time(), 0, 0
    while i < len(order):
        k = order[i]
        if n_tok[k] > chunk:
            acts = _forward_chunked(model, rendered[k][0], layers, chunk)
            results[k] = _pool(acts, rendered[k], W)
            i, done = i + 1, done + 1
        else:
            j = i + 1
            while j < len(order) and n_tok[order[j]] <= chunk and (j - i + 1) * n_tok[order[j]] <= token_budget:
                j += 1
            batch = order[i:j]
            width = max(n_tok[b] for b in batch)
            x = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for r, b in enumerate(batch):
                x[r, : n_tok[b]] = torch.tensor(rendered[b][0])
                mask[r, : n_tok[b]] = 1
            with capture_resid(model, layers) as cap:
                model.model(input_ids=x.to(model.device), attention_mask=mask.to(model.device), use_cache=False)
                for r, b in enumerate(batch):
                    results[b] = _pool({L: cap[L][r, : n_tok[b]] for L in layers}, rendered[b], W)
            done += len(batch)
            i = j
        if log_every and (done // log_every) != ((done - 1) // log_every):
            print(f"  re-read {done}/{len(convs)} conversations, {sum(n_tok[o] for o in order[:i]):,} tokens, {time.time() - t0:.0f}s", flush=True)
    out = {"ids": np.array([c["id"] for c in convs]), "n_tok": np.array(n_tok, np.int32), "truncated": np.array(truncated)}
    w_meta, w_conv, t_conv, t_turn = [], [], [], []
    for ci, (meta, _, turns, _, _) in enumerate(results):
        w_meta.append(meta)
        w_conv.append(np.full(len(meta), ci, np.int32))
        t_conv.append(np.full(len(turns), ci, np.int32))
        t_turn.append(turns)
    w_meta = np.concatenate(w_meta)
    out.update(w_conv=np.concatenate(w_conv), w_turn=w_meta[:, 0], w_seg=w_meta[:, 1], w_pos=w_meta[:, 2], w_len=w_meta[:, 3],
               t_conv=np.concatenate(t_conv), t_turn=np.concatenate(t_turn))
    for L in layers:
        out[f"W{L}"] = np.concatenate([r[1][L] for r in results])
        out[f"T{L}"] = np.concatenate([r[3][L] for r in results])
        out[f"norm{L}"] = np.array([r[4][L] for r in results], np.float32)
    out["seconds"] = np.array(time.time() - t0)
    return out
