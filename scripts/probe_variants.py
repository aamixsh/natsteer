"""(CPU; run from the repo root.) Probe variants on the same training data: is the window probe limited by label
noise inside misaligned turns, by mean pooling, or by being linear?

  probe_variants.py [--variants base topk mil energy fisher perbeh layers] [--tag w8 --out .../report_w8.json]

Training data: the three branch_sol runs (arms/branch_sol/s{2,3,4}/r0, gpt-6.1-sol labels, branch negatives), 32-token
window means at layer 15, every window labeled with its turn's label (as the deck's monitors). Variants:
  base     StandardScaler + logistic regression, C 3e-4 (reproduces the stored monitors)
  topk     multiple-instance relabeling: after a base fit, keep only the top 25% (topk25) or the top window (top1) of
           each misaligned turn as positives, drop its other windows, refit; two rounds
  mil      multiple-instance loss: a misaligned turn is one positive bag scored by a soft maximum of its window scores
           (log-mean-exp, sharpness 5); every aligned window is one negative; same L2 strength as base
  fisher   whitening by the training data's aligned windows (top 256 principal directions, Ledoit-Wolf), then the
           mean difference of misaligned windows (Fisher's discriminant)
  energy   the same whitening, then the top-k eigenvectors of the misaligned windows' second moment (directions with
           extra variance in misaligned turns); window score = energy in those k directions (k 1, 4, 16, 64)
  perbeh   one base probe per behavior (its misaligned windows vs all aligned windows), each z-scored on the training
           aligned windows; window score = maximum over the five
  layers   base on layers 11, 15 and 19 joined (8,640 features)
Tests (indicators convention: misaligned conversation vs aligned turn; turn score = max window): own rollouts from
held-out scenarios with gpt-6.1-sol labels and with the majority of gpt-6.1-sol and the two Qwen judges
(data/judges), Bloom test and val (Opus turn labels), Claude-written (indicators_s), own writes from held-out seeds.
--tag reads other window features (IN.<tag>.feats.npz from loop_features.py --tag; layer 15 only).
Output: docs/results/probe_variants_v0/report.json (mean, SD and per-run values) and a table (mean over the three runs).
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from natsteer.convo import load_jsonl
from natsteer.monitor import evaluate_scores, load, train, window_scores

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--variants", nargs="*", default=["base", "topk", "mil", "fisher", "energy", "perbeh", "layers"])
ap.add_argument("--C", type=float, default=3e-4)
ap.add_argument("--tag", default="", help="window features IN.<tag>.feats.npz (e.g. w8)")
ap.add_argument("--out", default="docs/results/probe_variants_v0/report.json")
args = ap.parse_args()

A, L = Path("data/loop/arms"), Path("data/loop")
RUNS = [A / "branch_sol" / f"s{k}" / "r0" for k in (2, 3, 4)]
LAYERS = (11, 15, 19)
KEYS = ("auroc", "tpr@1", "tpr@5")
need_layers = LAYERS if "layers" in args.variants else (15,)


# ------------------------------------------------------------------ test sets
def majority_labels():
    """Per-turn majority of gpt-6.1-sol and the two Qwen judges on the own rollouts (as scripts/judge_audit.py)."""
    sol = {c["id"]: c["turn_labels"] for c in load_jsonl(A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl")}
    q = [{r["id"]: r["turns"] for r in load_jsonl(Path("data/judges") / j / "onpolicy.jsonl") if r["turns"] is not None}
         for j in ("qwen3.6-35b-a3b", "qwen3.8-27b")]
    return {i: [int(a + b + c >= 2) for a, b, c in zip(t, q[0][i], q[1][i])] for i, t in sol.items() if i in q[0] and i in q[1]}


def load_layers(paths):
    """{layer: FeatureSet}; all share the window order."""
    return {lay: load(paths, lay, args.tag) for lay in need_layers}


seed_groups = {s["group"] for s in load_jsonl(L / "seeds_onpolicy.jsonl") if s["source"] == "onpolicy"}
tests = {n: load_layers([L / "eval" / f"{n}.jsonl"]) for n in ("bloom_test", "bloom_val", "indicators_s", "own")}
own = load_layers([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"])
maj = majority_labels()
own_maj = load_layers([A / "seedsrc" / "onpolicy_direct" / "r0" / "candidates.jsonl"])
for fsd, lab in ((own, None), (own_maj, maj)):
    for fs in fsd.values():
        for c in fs.convs:
            c["turn_labels"] = None if c["group"] in seed_groups else (lab.get(c["id"]) if lab is not None else c["turn_labels"])
tests["own_rollouts_sol"], tests["own_rollouts_maj"] = own, own_maj
print("tests loaded:", {n: len(t[15].convs) for n, t in tests.items()}, flush=True)


# ------------------------------------------------------------------ variants
def logit(X, y, C=args.C):
    m = train(X, y, C)
    return lambda Z: window_scores(m, Z)


def turn_index(fs, mask):
    """{(conv, turn): window indices} for windows in mask."""
    out = defaultdict(list)
    for i in np.where(mask)[0]:
        out[(int(fs.w_conv[i]), int(fs.w_turn[i]))].append(i)
    return out


def fit_topk(fs, frac):
    lab = ~np.isnan(fs.y)
    f = logit(fs.X[lab], fs.y[lab])
    for _ in range(2):
        s = f(fs.X)
        keep = lab & (fs.y == 0)
        for idx in turn_index(fs, fs.y == 1).values():
            idx = np.array(idx)
            n = 1 if frac is None else max(1, int(round(frac * len(idx))))
            keep[idx[np.argsort(-s[idx])[:n]]] = True
        f = logit(fs.X[keep], fs.y[keep])
    return f


def fit_mil(fs, sharp=5.0, iters=300):
    lab = ~np.isnan(fs.y)
    X = fs.X[lab].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.from_numpy((fs.X.astype(np.float32) - mu) / sd)
    neg = torch.from_numpy(np.where(lab & (fs.y == 0))[0])
    bags = list(turn_index(fs, fs.y == 1).values())
    bag_of = torch.from_numpy(np.concatenate([[k] * len(b) for k, b in enumerate(bags)]))
    pos = torch.from_numpy(np.concatenate(bags))
    lam = 1.0 / (2 * args.C * lab.sum())  # sklearn's balanced objective divided by C * n
    w = torch.zeros(Xs.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    nb = torch.bincount(bag_of).float()

    def closure():
        opt.zero_grad()
        sp = Xs[pos] @ w + b
        top = torch.zeros(len(bags)).scatter_reduce(0, bag_of, sp.detach(), reduce="amax", include_self=False)
        bag = torch.zeros(len(bags)).index_add(0, bag_of, torch.exp(sharp * (sp - top[bag_of]))).div(nb).log().div(sharp) + top
        sn = Xs[neg] @ w + b
        loss = 0.5 * (torch.nn.functional.softplus(-bag).mean() + torch.nn.functional.softplus(sn).mean()) + lam * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    wv, bv = w.detach().numpy(), float(b.detach())
    return lambda Z: ((Z.astype(np.float32) - mu) / sd) @ wv + bv


def whitener(fs, rank=256):
    neg = fs.X[fs.y == 0]
    sub = neg[np.random.default_rng(0).choice(len(neg), min(20000, len(neg)), replace=False)].astype(np.float64)
    mu = sub.mean(0)
    ev, V = np.linalg.eigh(LedoitWolf().fit(sub).covariance_)
    ev, V = ev[::-1][:rank], V[:, ::-1][:, :rank]
    W = (V / np.sqrt(ev)).astype(np.float32)
    return lambda Z: (Z.astype(np.float32) - mu.astype(np.float32)) @ W


def fit_fisher(fs):
    wh = whitener(fs)
    d = wh(fs.X[fs.y == 1]).mean(0)
    return lambda Z: wh(Z) @ d


def fit_energy(fs, ks=(1, 4, 16, 64)):
    wh = whitener(fs)
    Zp = wh(fs.X[fs.y == 1]).astype(np.float64)
    ev, U = np.linalg.eigh(Zp.T @ Zp / len(Zp))
    U = U[:, ::-1].astype(np.float32)
    print(f"    energy: top second-moment eigenvalues of misaligned windows (whitened; aligned = 1): {ev[::-1][:6].round(1)}", flush=True)
    return {k: (lambda Z, k=k: ((wh(Z) @ U[:, :k]) ** 2).sum(1)) for k in ks}


def fit_perbeh(fs):
    lab = ~np.isnan(fs.y)
    beh = np.array([fs.convs[c]["behavior"] for c in fs.w_conv])
    fns = []
    for bname in sorted(set(beh[fs.y == 1])):
        sel = lab & ((fs.y == 0) | (beh == bname))
        f = logit(fs.X[sel], fs.y[sel])
        sn = f(fs.X[lab & (fs.y == 0)])
        fns.append((f, sn.mean(), sn.std()))
    return lambda Z: np.max([(f(Z) - m) / s for f, m, s in fns], axis=0)


# ------------------------------------------------------------------ run
res = defaultdict(lambda: defaultdict(list))  # variant -> test -> [metrics per run]


joined = {}


def score(name, fn, layered=False):
    for tname, t in tests.items():
        if layered and tname not in joined:
            joined[tname] = np.hstack([t[lay].X for lay in LAYERS])
        Z = joined[tname] if layered else t[15].X
        e = evaluate_scores(t[15], fn(Z), n_boot=0)
        res[name][tname].append({k: e[k] for k in KEYS})


for d in RUNS:
    print(f"== {d}", flush=True)
    tr = load_layers([d / "candidates.jsonl"])
    fs = tr[15]
    lab = ~np.isnan(fs.y)
    if "base" in args.variants:
        f = logit(fs.X[lab], fs.y[lab])
        if not args.tag:  # the stored monitors are trained on the default 32-token windows
            stored = pickle.load(open(d / "metrics_c3e-4.monitor.pkl", "rb"))
            print(f"  base: max |score - stored monitor| on 100 windows {np.abs(f(fs.X[:100]) - window_scores(stored, fs.X[:100])).max():.2e}", flush=True)
        score("base", f)
    if "topk" in args.variants:
        score("topk25", fit_topk(fs, 0.25))
        score("top1", fit_topk(fs, None))
    if "mil" in args.variants:
        score("mil", fit_mil(fs))
    if "fisher" in args.variants:
        score("fisher", fit_fisher(fs))
    if "energy" in args.variants:
        for k, f in fit_energy(fs).items():
            score(f"energy{k}", f)
    if "perbeh" in args.variants:
        score("perbeh", fit_perbeh(fs))
    if "layers" in args.variants:
        Xc = np.hstack([tr[lay].X for lay in LAYERS])
        score("layers", logit(Xc[lab], fs.y[lab]), layered=True)
    print("  done", flush=True)

report = {v: {t: {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))] for k in KEYS} | {"runs": rs}
              for t, rs in ts.items()} for v, ts in res.items()}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=1))
cols = ["own_rollouts_sol", "own_rollouts_maj", "bloom_test", "bloom_val", "indicators_s", "own"]
print("\nmean over 3 runs: AUROC / detection at 1% FPR")
print(f"{'variant':10s} " + " ".join(f"{c:>20s}" for c in cols))
for v, ts in report.items():
    print(f"{v:10s} " + " ".join(f"{ts[c]['auroc'][0]:9.3f} / {ts[c]['tpr@1'][0]:.2f}  " for c in cols))
print(f"\nwrote {args.out}")
