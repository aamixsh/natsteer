"""All usable Aegis 2.0 train prompts with human labels, for human-label LASER baselines.

Writes data/pools/aegis_v0/train.jsonl, excluding prompts that appear in intent_v0's
held-out or human_eval files.
"""

import json

from natsteer.config import DATA_ROOT
from natsteer.data import load_aegis


def main():
    held = set()
    for name in ("heldout", "human_eval"):
        held |= {json.loads(l)["user"] for l in open(DATA_ROOT / "pools" / "intent_v0" / f"{name}.jsonl")}
    rows = [r for r in load_aegis("train") if r["user"] not in held]
    out = DATA_ROOT / "pools" / "aegis_v0"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), sum(r["human_label"] for r in rows), out)


if __name__ == "__main__":
    main()
