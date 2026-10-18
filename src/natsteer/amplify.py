"""Amplification step of the loop: choose the next round's seeds and derive style vectors from accepted data.

Seeds: per behavior, the accepted positives of this round whose out-of-fold monitor score is in the lowest
`miss_q` share (the monitor finds them hardest), then farthest-point sampling in activation space against the
current seeds, so new seeds are both missed and different. Old seeds keep a decayed weight.

Style vectors: per behavior, k-means on the accepted positives' misaligned-turn features (standardized, PCA);
per cluster, mean misaligned-turn feature minus the mean feature of the matched aligned rewrites of the same
conversations, in raw activation space at the monitor layer. Steering vector = alpha x median assistant-token
norm x unit vector. Each card lists the SAE latents whose decoder rows are closest to the vector.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from natsteer.laser import farthest_point


def conv_turn_features(fs, which: str = "pos"):
    """{conv index: mean turn feature} over misaligned turns ("pos") or all turns ("all")."""
    acc = {}
    for i, (c, t) in enumerate(zip(fs.t_conv, fs.t_turn)):
        tl = fs.convs[c].get("turn_labels") or []
        if which == "pos" and not (t < len(tl) and tl[t] == 1):
            continue
        acc.setdefault(int(c), []).append(fs.T[i].astype(np.float32))
    return {c: np.mean(v, 0) for c, v in acc.items()}


def select_seeds(fs, candidates, oof, seed_feats, n_new: int, miss_q: float = 0.3, seed: int = 0):
    """Indices (into fs.convs) of new seeds among `candidates` (one behavior)."""
    cand = [c for c in candidates if oof.get(c) is not None]
    if not cand:
        return []
    cut = np.quantile([oof[c] for c in cand], miss_q)
    missed = [c for c in cand if oof[c] <= cut] or cand
    feats = conv_turn_features(fs, "pos")
    missed = [c for c in missed if c in feats]
    if len(missed) <= n_new:
        return missed
    E = np.stack([feats[c] for c in missed] + list(seed_feats))
    E = StandardScaler().fit_transform(E)
    E = PCA(min(32, E.shape[0] - 1, E.shape[1])).fit_transform(E)
    labeled = list(range(len(missed), len(E)))
    picked = farthest_point(E, list(range(len(missed))), labeled, n_new, np.random.default_rng(seed))
    return [missed[i] for i in picked]


def style_vectors(fs, positives, k: int, alpha: float, behavior: str, round_: int, sae_dec=None, seed: int = 0):
    """Style vector cards for one behavior from accepted positives and their matched negatives."""
    pos_f = conv_turn_features(fs, "pos")
    all_f = conv_turn_features(fs, "all")
    by_id = {c["id"]: i for i, c in enumerate(fs.convs)}
    pairs = [(c, by_id.get(fs.convs[c]["meta"].get("pair"))) for c in positives if c in pos_f]
    pairs = [(c, n) for c, n in pairs if n is not None and n in all_f]
    if len(pairs) < 2 * k:
        return []
    P = np.stack([pos_f[c] for c, _ in pairs])
    N = np.stack([all_f[n] for _, n in pairs])
    Z = PCA(min(16, len(P) - 1)).fit_transform(StandardScaler().fit_transform(P))
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    norm = float(np.nanmedian(fs.norm)) if fs.norm is not None else 1.0
    cards = []
    for j in range(k):
        sel = lab == j
        if sel.sum() < 2:
            continue
        d = P[sel].mean(0) - N[sel].mean(0)
        u = d / np.linalg.norm(d)
        card = {"id": f"r{round_}:{behavior}:c{j}", "behavior": behavior, "round": round_, "layer": fs.layer, "alpha": alpha,
                "resid_norm": norm, "n": int(sel.sum()), "diff_norm": float(np.linalg.norm(d)),
                "examples": [fs.convs[pairs[i][0]]["id"] for i in np.flatnonzero(sel)[:5]], "vec": None}
        import torch

        card["vec"] = torch.tensor(alpha * norm * u, dtype=torch.float32)
        if sae_dec is not None:
            cos = sae_dec @ u / np.linalg.norm(sae_dec, axis=1).clip(1e-6)
            top = np.argsort(-cos)[:5]
            card["sae_latents"] = [(int(t), float(cos[t])) for t in top]
        cards.append(card)
    return cards
