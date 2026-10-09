"""LASER-style active learning (Etinger, alignment.openai.com/laser).

Each round: fit a logistic probe on the labeled set, score the unlabeled pool,
keep candidates whose predicted probability falls in an uncertainty band
(start [0.4, 0.6], widen up to [0.3, 0.7] until there are enough), then pick a
diverse batch by farthest-point sampling in a low-dimensional embedding.
`strategy="random"` gives the random-sampling baseline under the same budget, and
`strategy="uncertainty"` samples the band uniformly (LASER without farthest-point).
"""

from collections.abc import Callable

import numpy as np
from sklearn.decomposition import PCA

from natsteer.probes import auroc, make_probe, tpr_at_fpr


def uncertainty_candidates(probs, unlabeled, n, band=(0.4, 0.6), max_band=(0.3, 0.7), step=0.05):
    lo, hi = band
    while True:
        cand = unlabeled[(probs[unlabeled] >= lo) & (probs[unlabeled] <= hi)]
        if len(cand) >= n or (lo <= max_band[0] and hi >= max_band[1]):
            break
        lo, hi = max(lo - step, max_band[0]), min(hi + step, max_band[1])
    if len(cand) < n:
        # Fill from the closest-to-boundary remainder so every round spends its budget.
        rest = np.setdiff1d(unlabeled, cand)
        rest = rest[np.argsort(np.abs(probs[rest] - 0.5))][: n - len(cand)]
        cand = np.concatenate([cand, rest])
    return cand


def farthest_point(emb, cand, labeled, n, rng):
    """Greedy farthest-point selection of n items from cand, seeded by distance to labeled."""
    if len(cand) <= n:
        return cand
    ref = emb[labeled] if len(labeled) else emb[rng.choice(cand, 1)]
    c = emb[cand]
    c2 = (c**2).sum(-1)
    d = np.full(len(cand), np.inf)
    for i in range(0, len(ref), 1024):
        r = ref[i : i + 1024]
        d = np.minimum(d, (c2[:, None] + (r**2).sum(-1)[None, :] - 2 * c @ r.T).min(1))
    chosen = []
    for _ in range(n):
        j = int(np.argmax(d))
        chosen.append(cand[j])
        d = np.minimum(d, ((c - c[j]) ** 2).sum(-1))
        d[j] = -np.inf
    return np.asarray(chosen)


def run(
    X_pool: np.ndarray,
    oracle: Callable[[np.ndarray], np.ndarray],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    budget: int,
    batch: int = 50,
    init: int = 100,
    strategy: str = "laser",
    embed_dim: int = 256,
    fpr: float = 0.01,
    seed: int = 0,
) -> list[dict]:
    """Run one active-learning arm. oracle(indices) -> 0/1 labels (the grader).

    Returns the learning curve: metrics on (X_eval, y_eval) after each round.
    """
    rng = np.random.default_rng(seed)
    n = len(X_pool)
    labeled = rng.choice(n, min(init, budget), replace=False)
    y = {int(i): int(v) for i, v in zip(labeled, oracle(labeled))}
    emb = None
    if strategy == "laser":
        emb = PCA(min(embed_dim, X_pool.shape[1], n), random_state=seed).fit_transform(X_pool).astype(np.float32)
    curve = []
    while True:
        idx = np.fromiter(y.keys(), int)
        lab = np.fromiter(y.values(), int)
        point = {"n_labels": len(idx), "n_pos": int(lab.sum())}
        if 0 < lab.sum() < len(lab):
            probe = make_probe().fit(X_pool[idx], lab)
            s = probe.decision_function(X_eval)
            point |= {"auroc": auroc(y_eval, s), f"tpr@{fpr}": tpr_at_fpr(y_eval, s, fpr), "tpr@0.05": tpr_at_fpr(y_eval, s, 0.05)}
        else:
            probe = None
        curve.append(point)
        if len(idx) >= budget:
            return curve
        k = min(batch, budget - len(idx))
        unlabeled = np.setdiff1d(np.arange(n), idx)
        if strategy == "random" or probe is None:
            pick = rng.choice(unlabeled, k, replace=False)
        elif strategy in ("laser", "uncertainty"):
            probs = np.full(n, 0.5)
            probs[unlabeled] = probe.predict_proba(X_pool[unlabeled])[:, 1]
            cand = uncertainty_candidates(probs, unlabeled, k)
            pick = farthest_point(emb, cand, idx, k, rng) if strategy == "laser" else rng.choice(cand, k, replace=False)
        else:
            raise ValueError(strategy)
        y |= {int(i): int(v) for i, v in zip(pick, oracle(pick))}
