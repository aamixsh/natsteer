"""(CPU; run from the repo root.) Round-0 seeds with hand-checked seeds for some behaviors.

  curate_seeds.py [--ids configs/seeds_curated.json] [--out data/loop/seeds_r0_v2.jsonl]

configs/seeds_curated.json: {behavior: {"r0": [ids], "heldout": [ids], "note": text}}, ids of transcripts in the pool
half of the indicators data (data/loop/external.jsonl), or "<candidates.jsonl>#<id>" for a conversation from one of
our runs. A seed may come from another behavior's files (e.g. sabotage seeds from the self-preservation files: an
assistant that degrades the handover for its replacement); it gets the new behavior and meta.curated_from. Behaviors not in the file keep their seeds from data/loop/seeds_r0.jsonl and
seeds_heldout.jsonl. Writes <out> and the held-out file next to it (seeds_r0 -> seeds_heldout in the name). Use with
loop_generate.py --seeds <out>. The frozen data (seeds_r0.jsonl, external.jsonl, eval/) is not changed; the curated
seeds stay in external.jsonl, so an arm trained on external.jsonl has seen them.
"""

import argparse
import json
from pathlib import Path

from natsteer.convo import load_jsonl, save_jsonl

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ids", default="configs/seeds_curated.json")
ap.add_argument("--out", default="data/loop/seeds_r0_v2.jsonl")
args = ap.parse_args()

L = Path("data/loop")
cur = json.loads(Path(args.ids).read_text())
pool = {c["id"]: c for c in load_jsonl(L / "external.jsonl")}
out = {}
for split, old in (("r0", L / "seeds_r0.jsonl"), ("heldout", L / "seeds_heldout.jsonl")):
    rows = [s for s in load_jsonl(old) if s["behavior"] not in cur]
    for b, spec in cur.items():
        for i in spec[split]:
            path, _, cid = i.rpartition("#")
            c = next(x for x in load_jsonl(path) if x["id"] == cid) if path else pool[i]
            assert c["label"] == 1, i
            meta = c.get("meta", {}) | ({"curated_from": c["behavior"]} if c["behavior"] != b else {})
            rows.append(c | {"behavior": b, "meta": meta} | ({"weight": 1.0, "round": 0} if split == "r0" else {}))
    out[split] = rows
    print(split, {b: sum(r["behavior"] == b for r in rows) for b in sorted({r["behavior"] for r in rows})})
save_jsonl(Path(args.out), out["r0"])
save_jsonl(Path(args.out).with_name(Path(args.out).name.replace("seeds_r0", "seeds_heldout")), out["heldout"])
