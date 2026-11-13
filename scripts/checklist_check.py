"""(CPU + API; run from the repo root.) Does the checklist judge (natsteer.prompts.CHECKLIST, gpt-6.1-sol) agree with the
hand checks of the sabotage and sandbagging test writes?

  checklist_check.py [--cap-add 1.5] [--effort medium]

Items: the writes the per-turn judge accepted in arms/seedfix_smoke (v1), seedfix_v2_low, seedfix_v2_medium and
seedfix_v3, each read in full against fixed criteria (handcheck.json: S1-S6 for sabotage, B1-B5 for sandbagging; the
hand checks were done by a Claude subagent, not by people). A write passes the checklist if every item is 1.
Reported per behavior: precision and recall of a pass against the hand check, and per-criterion agreement (item ->
criterion: sabotage work S1, stake S2, not_instructed S3, flaw_in_reply and hidden S4, motive S5, scope S6;
sandbagging evaluation B1, not_instructed B2, error_in_reply and hidden B3, dual_process B4, scope B5; clean has no
hand criterion, so precision and recall are also given without it). Raw answers: data/loop/arms/checklist_check.jsonl
(skipped if present); report: docs/results/checklist_check_v0/report.json. Spend: the shared ledger, capped at its
total + --cap-add.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from natsteer.convo import load_jsonl, save_jsonl
from natsteer.remote import chat_many, ledger_total
from natsteer.writer import checklist_prompt, parse_checklist

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cap-add", type=float, default=1.5)
ap.add_argument("--effort", default="medium")
ap.add_argument("--out", default="docs/results/checklist_check_v0/report.json")
args = ap.parse_args()

A = Path("data/loop/arms")
RUNS = {"v1": "seedfix_smoke", "v2_low": "seedfix_v2_low", "v2_medium": "seedfix_v2_medium", "v3": "seedfix_v3"}
MAP = {"sabotage": {"work": ["S1"], "stake": ["S2"], "not_instructed": ["S3"], "flaw_in_reply": ["S4"], "hidden": ["S4"], "motive": ["S5"], "scope": ["S6"]},
       "sandbagging": {"evaluation": ["B1"], "not_instructed": ["B2"], "error_in_reply": ["B3"], "hidden": ["B3"], "dual_process": ["B4"], "scope": ["B5"]}}
LEDGER = A / "sol_spend.jsonl"
raw_path = A / "checklist_check.jsonl"

items = []
for ver, d in RUNS.items():
    hand = {r["id"]: r for r in json.load(open(A / d / "handcheck.json"))}
    for c in load_jsonl(A / d / "r0" / "candidates.jsonl"):
        if c["id"] in hand:
            items.append((ver, c, hand[c["id"]]))
print(f"{len(items)} hand-checked writes", flush=True)

if not raw_path.exists():
    res = chat_many([checklist_prompt(c["behavior"], c["messages"]) for _, c, _ in items], effort=args.effort, workers=16, ledger=LEDGER,
                    cap_usd=round(ledger_total(LEDGER) + args.cap_add, 2), tag="checklist_check")
    save_jsonl(raw_path, [{"ver": v, "id": c["id"], "behavior": c["behavior"], "text": r.get("text", r.get("error", "")), "cost": r.get("cost", 0.0)}
                          for (v, c, _), r in zip(items, res)])
raw = {(r["ver"], r["id"]): r for r in load_jsonl(raw_path)}
print(f"cost ${sum(r['cost'] for r in raw.values()):.2f}", flush=True)


def crit(h, keys):
    return all(next(v for k, v in h["pass"].items() if k.split("_")[0] == key) for key in keys)


report = {"n": len(items), "by_behavior": {}}
for b in ("sabotage", "sandbagging"):
    rows = []
    for ver, c, h in items:
        if c["behavior"] != b:
            continue
        ans = parse_checklist(raw[(ver, c["id"])]["text"], b)
        rows.append((ver, h, ans))
    ok = [(v, h, a) for v, h, a in rows if a is not None]
    out = {"n": len(rows), "unparsed": len(rows) - len(ok)}
    for name, keys in (("all_items", None), ("without_clean", [k for k in MAP[b]])):
        pred = np.array([all(a[k] for k in (keys or a)) for _, _, a in ok])
        hand = np.array([h["all_pass"] for _, h, _ in ok])
        tp = int((pred & hand).sum())
        out[name] = {"pass": int(pred.sum()), "hand_fits": int(hand.sum()), "both": tp,
                     "precision": tp / max(pred.sum(), 1), "recall": tp / max(hand.sum(), 1),
                     "per_version": {v: {"pass": int(sum(p for p, (vv, _, _) in zip(pred, ok) if vv == v)),
                                         "both": int(sum(p and hh for p, hh, (vv, _, _) in zip(pred, hand, ok) if vv == v)),
                                         "hand_fits": int(sum(hh for hh, (vv, _, _) in zip(hand, ok) if vv == v))} for v in RUNS}}
    agree = defaultdict(list)
    for _, h, a in ok:
        for k, keys in MAP[b].items():
            agree[k].append(int(a[k] == crit(h, keys)))
    out["item_agreement"] = {k: float(np.mean(v)) for k, v in agree.items()}
    out["item_pass_rate"] = {k: float(np.mean([a[k] for _, _, a in ok])) for k in ok[0][2]} if ok else {}
    report["by_behavior"][b] = out
    print(f"\n== {b}: {out['n']} writes ({out['unparsed']} unparsed), hand check fits {out['all_items']['hand_fits']}")
    for name in ("all_items", "without_clean"):
        o = out[name]
        print(f"  checklist ({name}): passes {o['pass']}, of them fit {o['both']} -> precision {o['precision']:.2f}, recall {o['recall']:.2f}; "
              f"per version (pass/both/fits) " + "  ".join(f"{v} {x['pass']}/{x['both']}/{x['hand_fits']}" for v, x in o["per_version"].items()))
    print("  item agreement with the hand criterion: " + "  ".join(f"{k} {v:.2f}" for k, v in out["item_agreement"].items()))
    print("  item pass rate: " + "  ".join(f"{k} {v:.2f}" for k, v in out["item_pass_rate"].items()))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(report, indent=1))
print(f"\nwrote {args.out}")
