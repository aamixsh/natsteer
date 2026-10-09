"""Grade jsonl prompt files with a policy. Writes <input>.grades.<policy>.<effort>.jsonl.

Usage: .venv/bin/python scripts/grade.py --policy configs/policies/X.md data/pools/intent_v0/*.jsonl
Resumes: ids already present in the output file are skipped.
"""

import argparse
import json
import time
from pathlib import Path

from natsteer.grader import DEFAULT_GRADER, PolicyGrader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--model", default=DEFAULT_GRADER)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--keep-cot", action="store_true")
    args = ap.parse_args()
    grader = PolicyGrader(args.policy, args.model, args.effort)
    for path in map(Path, args.inputs):
        out = path.with_name(f"{path.stem}.grades.{grader.policy_name}.{args.effort}.jsonl")
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        rows = [r for r in map(json.loads, open(path)) if r["id"] not in done][: args.limit]
        t0 = time.time()
        for i in range(0, len(rows), args.chunk):
            chunk = rows[i : i + args.chunk]
            res = grader.grade([r["user"] for r in chunk], keep_cot=args.keep_cot)
            with open(out, "a") as f:
                for r, g in zip(chunk, res):
                    f.write(json.dumps({"id": r["id"], "source": r["source"], **g}) + "\n")
            print(f"{path.name}: {i + len(chunk)}/{len(rows)} graded, {time.time() - t0:.0f}s", flush=True)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
