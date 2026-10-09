"""Pooled features, logistic probes, and fixed-FPR metrics."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POOLINGS = ("user_mean", "prompt_last", "cot_mean", "final_mean")


def pool(acts: torch.Tensor, labels: list[str], how: str) -> torch.Tensor:
    """Pool one sequence's activations [seq, d_model] using its segment labels."""
    lab = np.asarray(labels)
    if how == "user_mean":
        idx = np.flatnonzero(lab == "user")
    elif how == "cot_mean":
        idx = np.flatnonzero(lab == "cot")
    elif how == "final_mean":
        idx = np.flatnonzero(lab == "final")
    elif how == "prompt_last":
        # The token the model reads just before generating: `assistant` in
        # `<|end|><|start|>assistant`, three positions after the last user token.
        user = np.flatnonzero(lab == "user")
        idx = np.array([user[-1] + 3]) if len(user) and user[-1] + 3 < len(lab) else np.array([], dtype=int)
    else:
        raise ValueError(how)
    if len(idx) == 0:
        return torch.full((acts.shape[-1],), float("nan"))
    return acts[idx].float().mean(0)


def make_probe(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def oof_scores(X: np.ndarray, y: np.ndarray, folds: int = 5, C: float = 1.0, seed: int = 0) -> np.ndarray:
    """Out-of-fold probe scores for a quick cross-validated estimate."""
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        out[te] = make_probe(C).fit(X[tr], y[tr]).decision_function(X[te])
    return out


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false positive rate (score > thr) is at most fpr."""
    neg = np.sort(scores[y == 0])
    k = int(np.floor(fpr * len(neg)))  # number of negatives allowed above the threshold
    return neg[len(neg) - k - 1] if k < len(neg) else -np.inf


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    thr = threshold_at_fpr(y, scores, fpr)
    return float(np.mean(scores[y == 1] > thr))


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))


def bootstrap_ci(metric, y: np.ndarray, scores: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Stratified bootstrap interval, resampling positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(metric(y[idx], scores[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
