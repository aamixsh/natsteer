"""Probe evaluation for synthetic-versus-natural training data.

- run_arm_grid: fits probes for every (arm, label budget, seed) in parallel and scores them on natural
  held-out rows, with metrics (AUROC, TPR at 1% and 5% FPR) on several slices and prompt-clustered
  bootstrap intervals (rows of one prompt are resampled together). Differences against a reference arm use
  the same resamples (paired).
- precision_recall: improved precision/recall (kNN balls) of a synthetic set against natural data, so
  naturalness (precision) and coverage/diversity (recall) are separate numbers.
- matched_discriminator: synthetic-vs-natural discriminator AUC after matching every synthetic sample to
  a natural sample with the nearest prompt, so topic cannot carry the separation.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from natsteer.probes import make_probe

METRICS = ("auroc", "tpr@0.01", "tpr@0.05")
_CTX: dict = {}


def tpr_np(y, s, fpr):
    neg = s[y == 0]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return np.nan
    k = int(np.floor(fpr * len(neg)))
    thr = np.partition(neg, len(neg) - k - 1)[len(neg) - k - 1] if k < len(neg) else -np.inf
    return float(np.mean(s[y == 1] > thr))


def auroc_np(y, s):
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metric_vec(y, s):
    return np.array([auroc_np(y, s), tpr_np(y, s, 0.01), tpr_np(y, s, 0.05)])


def make_resamples(groups, n_boot, seed=0):
    """Index arrays that resample whole groups (prompts) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([rows[g] for g in pick]))
    return out


def _job(args):
    arm, b, seed = args
    from threadpoolctl import threadpool_limits

    c = _CTX
    with threadpool_limits(1):
        tr = c["arms"][arm](b, seed)
        if tr is None:
            return args, None
        X, y = tr
        if len(set(y.tolist())) < 2:
            return args, None
        if c.get("project") is not None:
            u = c["project"]
            X = X - np.outer(X @ u, u)
            Xe = c["Xev"] - np.outer(c["Xev"] @ u, u)
        else:
            Xe = c["Xev"]
        s = make_probe().fit(X, y).decision_function(Xe)
    yev = c["yev"]
    point = np.zeros((len(c["slices"]), 3))
    boot = np.full((c["n_boot"], len(c["slices"]), 3), np.nan)
    for j, (name, mask) in enumerate(c["slices"].items()):
        idx = np.flatnonzero(mask)
        point[j] = metric_vec(yev[idx], s[idx])
        for r, rs in enumerate(c["resamples"][name]):
            ii = idx[rs]
            boot[r, j] = metric_vec(yev[ii], s[ii])
    return args, (point, boot)


def run_arm_grid(arms, Xev, yev, groups_ev, slices, budgets, seeds, n_boot=200, n_jobs=32, project=None, reference="natural"):
    """arms: {name: fn(budget, seed) -> (X_train, y_train) or None}. Returns {slice: {budget: {arm: stats}}}.

    stats = {metric: {"mean", "lo", "hi", "n_seeds"}} plus, for arms other than `reference`,
    {"delta_<metric>": {"mean", "lo", "hi"}} against the reference arm at the same budget and resamples.
    """
    _CTX.clear()
    resamples = {name: make_resamples(groups_ev[np.flatnonzero(m)], n_boot, seed=1) for name, m in slices.items()}
    _CTX.update(arms=arms, Xev=Xev, yev=yev, slices=slices, resamples=resamples, n_boot=n_boot, project=project)
    jobs = [(a, b, s) for a in arms for b in budgets for s in range(seeds)]
    res = {}
    with ProcessPoolExecutor(max_workers=min(n_jobs, os.cpu_count() or 1), mp_context=__import__("multiprocessing").get_context("fork")) as ex:
        for args, out in ex.map(_job, jobs, chunksize=4):
            if out is not None:
                res[args] = out
    out = {sl: {} for sl in slices}
    slice_names = list(slices)
    for b in budgets:
        series = {}
        for a in arms:
            got = [res[(a, b, s)] for s in range(seeds) if (a, b, s) in res]
            if got:
                series[a] = (np.mean([g[0] for g in got], 0), np.nanmean([g[1] for g in got], 0), len(got))
        for j, sl in enumerate(slice_names):
            entry = {}
            for a, (pt, bt, n) in series.items():
                st = {}
                for m, mname in enumerate(METRICS):
                    lo, hi = np.nanpercentile(bt[:, j, m], [2.5, 97.5])
                    st[mname] = {"mean": float(pt[j, m]), "lo": float(lo), "hi": float(hi), "n_seeds": n}
                if reference in series and a != reference:
                    rp, rb, _ = series[reference]
                    for m, mname in enumerate(METRICS):
                        d = bt[:, j, m] - rb[:, j, m]
                        lo, hi = np.nanpercentile(d, [2.5, 97.5])
                        st["delta_" + mname] = {"mean": float(pt[j, m] - rp[j, m]), "lo": float(lo), "hi": float(hi)}
                entry[a] = st
            out[sl][str(b)] = entry
    return out


def precision_recall(real, fake, k=3, n_comp=50, max_n=1500, seed=0):
    """Improved precision/recall (Kynkaanniemi et al.): precision = share of fake points inside the kNN
    balls of real points, recall = share of real points inside the kNN balls of fake points. Features are
    standardized and reduced with PCA fit on the real set."""
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), min(len(real), max_n), replace=False)]
    fake = fake[rng.choice(len(fake), min(len(fake), max_n), replace=False)]
    mu, sd = real.mean(0), real.std(0) + 1e-6
    pca = PCA(min(n_comp, len(real) - 1), random_state=0).fit((real - mu) / sd)
    R, F = pca.transform((real - mu) / sd), pca.transform((fake - mu) / sd)

    def radii(P):
        d = cdist(P, P)
        return np.sort(d, 1)[:, k]

    rr, fr = radii(R), radii(F)
    d_fr = cdist(F, R)
    precision = float(np.mean((d_fr <= rr[None, :]).any(1)))
    recall = float(np.mean((d_fr.T <= fr[None, :]).any(1)))
    return {"precision": precision, "recall": recall, "n_real": len(R), "n_fake": len(F)}


def matched_discriminator(Xs, Gs, Ps, Xn, Gn, Pn, max_n=1000, n_comp=30, seed=0):
    """Discriminator AUC (grouped CV by prompt) between synthetic (Xs) and natural (Xn) rows after
    matching each synthetic row to a distinct natural row with the nearest prompt feature (Ps, Pn).
    Also returns the AUC of a prompt-feature-only discriminator on the matched pairs (about 0.5 when
    the matching removed topic)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    si = rng.choice(len(Xs), min(len(Xs), max_n), replace=False)
    ni = rng.choice(len(Xn), min(len(Xn), 4 * max_n), replace=False)
    mu, sd = Pn[ni].mean(0), Pn[ni].std(0) + 1e-6
    pca = PCA(min(n_comp, len(ni) - 1), random_state=0).fit((Pn[ni] - mu) / sd)
    ps, pn = pca.transform((Ps[si] - mu) / sd), pca.transform((Pn[ni] - mu) / sd)
    rows, cols = linear_sum_assignment(cdist(ps, pn))
    S, N = si[rows], ni[cols]
    X = np.concatenate([Xs[S], Xn[N]])
    y = np.r_[np.ones(len(S)), np.zeros(len(N))].astype(int)
    g = np.concatenate([Gs[S], Gn[N]])
    P = np.concatenate([ps[rows], pn[cols]])

    def cv_auc(F):
        out = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F, y, g):
            out[te] = make_probe().fit(F[tr], y[tr]).decision_function(F[te])
        return float(roc_auc_score(y, out))

    return {"auc": cv_auc(X), "prompt_only_auc": cv_auc(P), "n_pairs": int(len(S))}
