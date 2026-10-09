"""Build the natural pool, held-out set and human-labeled eval set for the intent label.

Writes data/pools/intent_v0/{pool,heldout,human_eval}.jsonl. Seeded and deterministic.
Mostly WildChat traffic, plus a small enrichment slice from Aegis, red-team attempts
and HarmfulQA so positives are not vanishingly rare. Every row keeps its source.
"""

import json
import random
from pathlib import Path

from natsteer.config import DATA_ROOT
from natsteer.data import load_aegis, load_harmfulqa, load_redteam, load_wildchat, load_xstest

OUT = DATA_ROOT / "pools" / "intent_v0"
SIZES = {  # source: (pool, heldout)
    "wildchat": (30000, 5000),
    "aegis": (1500, 300),
    "redteam": (1000, 200),
    "harmfulqa": (500, 100),
}


def write(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    rng = random.Random(0)
    sources = {"wildchat": load_wildchat(), "aegis": load_aegis("train"), "redteam": load_redteam(), "harmfulqa": load_harmfulqa()}
    pool, heldout, seen = [], [], set()
    for name, (n_pool, n_held) in SIZES.items():
        rows = [r for r in sources[name] if r["user"] not in seen]
        rng.shuffle(rows)
        rows = rows[: n_pool + n_held]
        seen.update(r["user"] for r in rows)
        pool += rows[:n_pool]
        heldout += rows[n_pool:]
    human = [r for r in load_aegis("test") if r["user"] not in seen]
    human += [{"id": f"xstest:{r['id']}", "source": "xstest", "user": r["user"], "human_label": r["label"], "category": r["type"]} for r in load_xstest()]
    rng.shuffle(pool)
    rng.shuffle(heldout)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "pool.jsonl", pool)
    write(OUT / "heldout.jsonl", heldout)
    write(OUT / "human_eval.jsonl", human)
    print({"pool": len(pool), "heldout": len(heldout), "human_eval": len(human), "out": str(OUT)})


if __name__ == "__main__":
    main()
