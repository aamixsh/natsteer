"""Report for scripts/style_sweep.sh: what each style-vector strength does to the writer.

Per scale: share of writes in the right format, accepted positives per prompt, mean realism of accepted
positives, share of branch negatives accepted, repetition (share of repeated word 4-grams in assistant text; text
degradation), shift of misaligned-turn features along each behavior's vector in SDs of the unsteered (scale 0)
turns, and spread (mean pairwise distance of misaligned-turn features within a behavior, standardized space).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from natsteer.amplify import conv_turn_features
from natsteer.config import DATA_ROOT
from natsteer.monitor import load

O = DATA_ROOT / "loop" / "style_sweep"
cards = {c["behavior"]: c for c in torch.load(O / "vectors.pt")}
scales = sorted((float(p.name[1:]), p) for p in O.glob("s*") if (p / "candidates.feats.npz").exists())


def rep4(text):
    w = text.split()
    g = [tuple(w[i : i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


rows, feats = {}, {}
for sc, d in scales:
    st = json.loads((d / "stats.json").read_text())
    tot = Counter()
    for v in st["by_behavior_config"].values():
        tot.update(v)
    fs = load([d / "candidates.jsonl"], 15)
    pf = conv_turn_features(fs, "pos")
    by_b = defaultdict(list)
    for c, v in pf.items():
        by_b[fs.convs[c]["behavior"]].append(v)
    feats[sc] = by_b
    pos = [c for c in fs.convs if c["label"] == 1]
    rows[sc] = {"prompts": tot["prompts"], "format_ok": tot["write_ok"] / max(tot["prompts"], 1), "pos_per_prompt": tot["accepted_pos"] / max(tot["prompts"], 1),
                "neg_per_pos": tot["accepted_neg"] / max(tot["accepted_pos"], 1),
                "realism": float(np.mean([c["meta"]["realism"] for c in pos])) if pos else None,
                "repetition": float(np.mean([rep4(" ".join(m["content"] + " " + (m["reasoning"] or "") for m in c["messages"] if m["role"] == "assistant")) for c in pos])) if pos else None}
base = feats.get(0.0, {})
for sc, by_b in feats.items():
    shift, spread = {}, {}
    for b, vs in by_b.items():
        V = np.stack(vs)
        if b in cards and len(base.get(b, [])) >= 3:
            u = cards[b]["vec"].numpy() / np.linalg.norm(cards[b]["vec"].numpy())
            B = np.stack(base[b])
            shift[b] = float(((V - B.mean(0)) @ u).mean() / max(np.std(B @ u), 1e-6))
        if len(V) >= 3:
            Z = (V - V.mean(0)) / (np.concatenate([np.stack(x) for x in feats[0.0].values()]).std(0) + 1e-6) if base else V
            dd = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
            spread[b] = float(dd[np.triu_indices(len(V), 1)].mean())
    rows[sc]["shift_sd"] = shift
    rows[sc]["shift_sd_mean"] = float(np.mean(list(shift.values()))) if shift else None
    rows[sc]["spread_mean"] = float(np.mean(list(spread.values()))) if spread else None
(O / "report.json").write_text(json.dumps({"vectors": {b: c["id"] for b, c in cards.items()}, "by_scale": rows}, indent=1))
print(f"{'scale':>5} {'alpha':>5} {'format':>6} {'pos/prompt':>10} {'neg/pos':>7} {'realism':>7} {'repeat':>6} {'shift SD':>8} {'spread':>7}")
for sc in sorted(rows):
    r = rows[sc]
    f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    print(f"{sc:5.0f} {0.1 * sc:5.2f} {f(r['format_ok']):>6} {f(r['pos_per_prompt']):>10} {f(r['neg_per_pos']):>7} {f(r['realism']):>7} {f(r['repetition'], 3):>6} {f(r['shift_sd_mean']):>8} {f(r['spread_mean'], 1):>7}")
