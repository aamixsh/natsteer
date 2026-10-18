"""Train, evaluate and amplify after one round of an arm (CPU).

  loop_round.py --arm-dir data/loop/arms/full --round 0 --amplify full

Reads <arm-dir>/r0 .. r<round>/candidates.jsonl with their .feats.npz (training data so far), trains monitor
P_{round+1} on all of it, evaluates it on the frozen sets in data/loop/eval/ (each needs its .feats.npz), and
writes <arm-dir>/r<round>/monitor.pkl and metrics.json. With --amplify seeds|full it also writes the next round's
seeds (<arm-dir>/seeds_r<round+1>.jsonl) and, with full, style vector cards (<arm-dir>/vectors_r<round+1>.pt).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from natsteer.config import DATA_ROOT
from natsteer.convo import load_jsonl, save_jsonl
from natsteer.monitor import evaluate, load, oof_conv_scores, train

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--arm-dir", required=True)
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--amplify", choices=["none", "seeds", "full"], default="none")
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--eval", nargs="*", default=["own", "indicators_s", "bloom_val", "bloom_test"])
ap.add_argument("--eval-dir", default=str(DATA_ROOT / "loop" / "eval"))
ap.add_argument("--n-new-seeds", type=int, default=4)
ap.add_argument("--seed-decay", type=float, default=0.5)
ap.add_argument("--k-styles", type=int, default=3)
ap.add_argument("--style-alpha", type=float, default=0.1)
ap.add_argument("--seed0", default=str(DATA_ROOT / "loop" / "seeds_r0.jsonl"))
ap.add_argument("--no-sae", action="store_true")
ap.add_argument("--metrics-name", default="metrics.json", help="re-scoring: write metrics here (with --amplify none)")
args = ap.parse_args()

arm = Path(args.arm_dir)
rdir = arm / f"r{args.round}"
paths = [arm / f"r{r}" / "candidates.jsonl" for r in range(args.round + 1)]
fs = load(paths, args.layer)
lab = ~np.isnan(fs.y)
print(f"training data: {len(fs.convs)} conversations, {int(lab.sum())} labeled windows ({int((fs.y == 1).sum())} positive)", flush=True)
m = train(fs.X[lab], fs.y[lab], args.C)
pickle.dump(m, open(rdir / args.metrics_name.replace(".json", ".monitor.pkl").replace("metrics.monitor", "monitor"), "wb"))

metrics = {"round": args.round, "layer": args.layer, "C": args.C, "train_convs": len(fs.convs),
           "train_pos": sum(c["label"] == 1 for c in fs.convs), "train_neg": sum(c["label"] == 0 for c in fs.convs), "eval": {}}
for name in args.eval:
    p = Path(args.eval_dir) / f"{name}.jsonl"
    if not p.with_suffix(".feats.npz").exists():
        print(f"skip eval {name}: no features")
        continue
    metrics["eval"][name] = evaluate(load([p], args.layer), m)
    e = metrics["eval"][name]
    print(f"  {name:13s} AUROC {e['auroc']:.3f} [{e['auroc_ci'][0]:.3f}, {e['auroc_ci'][1]:.3f}]  TPR@1% {e['tpr@1']:.3f}  TPR@5% {e['tpr@5']:.3f}  "
          f"(pos {e['n_pos']}, neg turns {e['n_neg_turns']})", flush=True)

if args.amplify != "none":
    from natsteer.amplify import conv_turn_features, select_seeds, style_vectors

    start = len(fs.convs) - len(load_jsonl(paths[-1]))
    this_round = list(range(start, len(fs.convs)))
    oof = oof_conv_scores(fs, lab)
    prev_seeds = load_jsonl(arm / f"seeds_r{args.round}.jsonl") if (arm / f"seeds_r{args.round}.jsonl").exists() else load_jsonl(args.seed0)
    by_id = {c["id"]: i for i, c in enumerate(fs.convs)}
    pos_f = conv_turn_features(fs, "pos")
    seed0_feats = {}
    s0 = Path(args.seed0)
    if s0.with_suffix(".feats.npz").exists():
        fs0 = load([s0], args.layer)
        seed0_feats = {fs0.convs[c]["id"]: v for c, v in conv_turn_features(fs0, "pos").items()}
    new_seeds, cards, report = [], [], {}
    sae_dec = None
    if args.amplify == "full" and not args.no_sae:
        from natsteer.sae import load_andyrdt_sae

        sae_dec = load_andyrdt_sae(args.layer, 64, device="cpu").W_dec.detach().float().numpy()
        if sae_dec.shape[0] == fs.X.shape[1]:
            sae_dec = sae_dec.T
    behaviors = sorted({c["behavior"] for c in fs.convs})
    for b in behaviors:
        cand = [c for c in this_round if fs.convs[c]["behavior"] == b and fs.convs[c]["label"] == 1]
        old = [s for s in prev_seeds if s["behavior"] == b]
        sf = [pos_f[by_id[s["id"]]] if s["id"] in by_id and by_id[s["id"]] in pos_f else seed0_feats.get(s["id"]) for s in old]
        sf = [v for v in sf if v is not None]
        picked = select_seeds(fs, cand, oof, sf, args.n_new_seeds)
        for s in old:
            new_seeds.append(s | {"weight": s.get("weight", 1.0) * args.seed_decay})
        for c in picked:
            new_seeds.append(fs.convs[c] | {"weight": 1.0, "round": args.round + 1, "oof_score": float(oof[c])})
        report[b] = {"candidates": len(cand), "picked": [fs.convs[c]["id"] for c in picked],
                     "oof_median_candidates": float(np.median([oof[c] for c in cand if oof.get(c) is not None])) if cand else None,
                     "oof_picked": [float(oof[c]) for c in picked]}
        if args.amplify == "full":
            pos_all = [c for c in range(len(fs.convs)) if fs.convs[c]["behavior"] == b and fs.convs[c]["label"] == 1]
            bc = style_vectors(fs, pos_all, args.k_styles, args.style_alpha, b, args.round + 1, sae_dec)
            cards += bc
            report[b]["styles"] = [{k: v for k, v in c.items() if k != "vec"} for c in bc]
    save_jsonl(arm / f"seeds_r{args.round + 1}.jsonl", new_seeds)
    metrics["amplify"] = report
    if args.amplify == "full":
        import torch

        torch.save(cards, arm / f"vectors_r{args.round + 1}.pt")
    print(f"amplify: {sum(len(r['picked']) for r in report.values())} new seeds, {len(cards)} style vectors", flush=True)

(rdir / args.metrics_name).write_text(json.dumps(metrics, indent=1))
