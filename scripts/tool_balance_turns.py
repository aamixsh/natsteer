"""(CPU) Tool use by turn label in writer data: the share of misaligned and of aligned assistant turns that call a
tool, per arm, over agentic writes and their negatives (a gap = a tool shortcut for the monitor). Compares arms
before the branch fix (agentic_sol) and after (scale_v1). Output: docs/results/eval_audit_v0/tool_balance.json"""

import json
from collections import Counter
from pathlib import Path

from natsteer.convo import load_jsonl, message_turns

A = Path("data/loop/arms")
out = {}
for arm in ("agentic_sol", "scale_v1"):
    c = Counter()
    for p in sorted(A.glob(f"{arm}/s*/r0/candidates.jsonl")):
        for conv in load_jsonl(p):
            if not (conv.get("meta", {}).get("setting") or {}).get("tools"):
                continue
            tools = {t for m, t in zip(conv["messages"], message_turns(conv)) if m["role"] == "assistant" and m["content"].startswith("[tool call]")}
            for t, lab in enumerate(conv.get("turn_labels") or []):
                if lab is None:
                    continue
                c[f"{'mis' if lab else 'ali'}_turns"] += 1
                c[f"{'mis' if lab else 'ali'}_tool"] += t in tools
    if c:
        out[arm] = dict(c) | {"misaligned_with_tool": c["mis_tool"] / max(c["mis_turns"], 1), "aligned_with_tool": c["ali_tool"] / max(c["ali_turns"], 1)}
Path("docs/results/eval_audit_v0").mkdir(parents=True, exist_ok=True)
Path("docs/results/eval_audit_v0/tool_balance.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
