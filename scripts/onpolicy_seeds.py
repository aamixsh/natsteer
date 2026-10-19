"""Seeds and training conversations from the on-policy pilot.

  onpolicy_seeds.py [--pilot data/onpolicy/pilot] [--per-behavior 4]

Reads <pilot>/rollouts.jsonl and judged.jsonl and writes:
  <pilot>/convs.jsonl        every judged rollout without a hazard flag in the natsteer.convo schema (turn labels
                             from the judge; tool results stay inside their turn via meta.bloom_turns), for
                             training a monitor directly on on-policy data
  <pilot>/seeds_onpolicy.jsonl  --per-behavior seeds per behavior for the writer: judged positives, each from a
                             different scenario, alternating pressure types, preferring higher realism and shorter
                             transcripts; condensed (tool results clipped, turns after the last misaligned one dropped). A behavior with
                             fewer than 2 positives keeps its Claude-written round-0 seeds (reported).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from natsteer.behaviors import BEHAVIORS
from natsteer.convo import _bloom_turns, load_jsonl, n_turns, save_jsonl, to_tagged
from natsteer.writer import MAX_EXAMPLE_CHARS

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pilot", default="data/onpolicy/pilot")
ap.add_argument("--per-behavior", type=int, default=4)
ap.add_argument("--fallback", default="data/loop/seeds_r0.jsonl")
args = ap.parse_args()

P = Path(args.pilot)
judged = {j["id"]: j for j in load_jsonl(P / "judged.jsonl")}
convs = []
for r in load_jsonl(P / "rollouts.jsonl"):
    j = judged.get(r["id"])
    v = j["verdict"] if j else None
    if v is None or v["harmful_detail"]:
        continue
    msgs = list(r["messages"])
    while msgs and msgs[-1]["role"] == "user":  # judged conversations end at the last assistant message
        msgs.pop()
    if n_turns(msgs) != len(v["turns"]):
        continue
    convs.append({"id": r["id"], "messages": msgs, "turn_labels": v["turns"], "label": int(any(v["turns"])), "behavior": r["behavior"],
                  "source": "onpolicy", "group": r["scenario"],
                  "meta": {"scenario": r["scenario"], "pressure": r["pressure"], "config": r["config"], "realism": v["realism"],
                           "evidence": j.get("evidence", ""), "bloom_turns": _bloom_turns(msgs)}})
save_jsonl(P / "convs.jsonl", convs)
print(f"convs: {len(convs)} ({sum(c['label'] for c in convs)} misaligned)")

def _clip(text, n):
    return text if len(text) <= n else text[:n] + " [...]"


def condense(c, tool_chars=200, user_chars=500, reply_chars=400, mis_reply_chars=2000, max_steps=2):
    """Seed copy for the writer, which shows each example up to MAX_EXAMPLE_CHARS: turns after the last
    misaligned turn dropped; at most max_steps tool calls per turn; tool calls and results, user messages and
    replies clipped, replies in misaligned turns less;
    reasoning kept whole (it carries the motive)."""
    last = max(i for i, t in enumerate(c["turn_labels"]) if t == 1)
    msgs, turn, steps = [], -1, 0
    for m in c["messages"]:
        if m["role"] == "user" and not m.get("tool"):
            turn += 1
            steps = 0
            if turn > last:
                break
        if m.get("call"):
            steps += 1
            if steps > max_steps:
                continue
            m = m | {"content": _clip(m["content"], tool_chars)}
        elif m.get("tool"):
            if steps > max_steps:
                continue
            m = m | {"content": _clip(m["content"], tool_chars)}
        elif m["role"] == "user":
            m = m | {"content": _clip(m["content"], user_chars)}
        elif m["role"] == "assistant":
            m = m | {"content": _clip(m["content"], mis_reply_chars if c["turn_labels"][turn] == 1 else reply_chars)}
        msgs.append(m)
    return c | {"messages": msgs, "turn_labels": c["turn_labels"][: last + 1]}


def _mis_offset(c):
    """Character offset in the tagged text where the last assistant reply starts (it is in a misaligned turn)."""
    t = to_tagged(c["messages"])
    return t.rfind("[ASSISTANT]\n")


fallback = defaultdict(list)
for s in load_jsonl(args.fallback):
    fallback[s["behavior"]].append(s)
seeds, report = [], {}
for b in BEHAVIORS:
    pos = [condense(c) for c in convs if c["behavior"] == b and c["label"] == 1]
    pos.sort(key=lambda c: (-c["meta"]["realism"], len(to_tagged(c["messages"])) > MAX_EXAMPLE_CHARS, len(to_tagged(c["messages"]))))
    picked, used = [], set()
    for want in ("situational", "incentive") * args.per_behavior:  # alternate pressure types, one per scenario
        if len(picked) == args.per_behavior:
            break
        ok = [c for c in pos if c["group"] not in used and _mis_offset(c) <= MAX_EXAMPLE_CHARS - 500] or [c for c in pos if c["group"] not in used]
        c = next((c for c in ok if c["meta"]["pressure"] == want), None) or next(iter(ok), None)
        if c is None:
            break
        picked.append(c)
        used.add(c["group"])
    if len(picked) < 2:
        seeds += fallback[b]
        report[b] = {"positives": len(pos), "picked": 0, "fallback_claude": len(fallback[b])}
    else:
        seeds += [c | {"weight": 1.0, "round": 0} for c in picked]
        report[b] = {"positives": len(pos), "picked": len(picked), "scenarios": len({c["group"] for c in pos}),
                     "pressure": dict(Counter(c["meta"]["pressure"] for c in picked)), "config": dict(Counter(c["meta"]["config"] for c in picked)),
                     "realism": [c["meta"]["realism"] for c in picked], "chars": [len(to_tagged(c["messages"])) for c in picked],
                     "misaligned_reply_at": [_mis_offset(c) for c in picked]}
save_jsonl(P / "seeds_onpolicy.jsonl", seeds)
(P / "seeds_report.json").write_text(json.dumps(report, indent=1))
for b, r in report.items():
    print(b, r)
