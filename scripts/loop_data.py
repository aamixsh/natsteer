"""Frozen data for the self-generation loop: seeds, the external-writer pool and the evaluation sets.

Writes data/loop/:
  seeds_r0.jsonl        round-0 seeds: --n-seeds positives per behavior from the misalignment-indicators
                        training data (Claude Sonnet), hazard domains excluded
  seeds_heldout.jsonl   a disjoint seed set, used once to generate the level-1 test set (own generations)
  external.jsonl        indicators transcripts for the external-writer arm (half of each file, by id hash)
  eval/indicators.jsonl the other half (level 2: another writer)
  eval/bloom_val.jsonl, eval/bloom_test.jsonl
                        Bloom rollouts of GLM-4.7-Flash and Qwen3.6-35B-A3B with Opus turn labels (level 3);
                        the test split is downloaded from kzhou35/misalignment-indicators-bloom-rollouts
No step of the loop reads eval/.
"""

import argparse
import hashlib
import random
from collections import Counter

from natsteer.behaviors import BEHAVIORS, BLOOM_TEST, BLOOM_VAL, INDICATORS_DATA, bloom_folders, is_hazard_domain
from natsteer.config import DATA_ROOT
from natsteer.convo import load_bloom, load_indicators, save_jsonl

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(DATA_ROOT / "loop"))
ap.add_argument("--n-seeds", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-download", action="store_true")
args = ap.parse_args()

from pathlib import Path  # noqa: E402

out = Path(args.out)
rng = random.Random(args.seed)


def half(cid: str) -> int:
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 2


def usable_seed(c) -> bool:
    asst = [m for m in c["messages"] if m["role"] == "assistant"]
    return c["label"] == 1 and len(asst) >= 2 and all(m.get("reasoning") for m in asst) and not is_hazard_domain(c)


seeds_r0, seeds_ho, external, ind_eval = [], [], [], []
for b, card in BEHAVIORS.items():
    convs = []
    for f in card["indicators"]:
        convs += load_indicators(INDICATORS_DATA / f"{f}.json", behavior=b)
    pool = [c for c in convs if half(c["id"]) == 0]
    test = [c for c in convs if half(c["id"]) == 1]
    cand = [c for c in pool if usable_seed(c)]
    rng.shuffle(cand)
    s0, sh = cand[: args.n_seeds], cand[args.n_seeds : 2 * args.n_seeds]
    used = {c["id"] for c in s0 + sh}
    for c in s0:
        seeds_r0.append(c | {"weight": 1.0, "round": 0})
    seeds_ho += sh
    external += [c for c in pool if c["id"] not in used]
    ind_eval += test
    print(f"{b}: {len(convs)} indicators transcripts, {len(cand)} usable seeds, external {len(pool) - len(used)}, eval {len(test)}")
save_jsonl(out / "seeds_r0.jsonl", seeds_r0)
save_jsonl(out / "seeds_heldout.jsonl", seeds_ho)
save_jsonl(out / "external.jsonl", external)
save_jsonl(out / "eval" / "indicators.jsonl", ind_eval)

if not args.skip_download:
    from huggingface_hub import snapshot_download

    pats = []
    for card in BEHAVIORS.values():
        for m in ("glm_4_7_flash", "qwen3_6_35b_a3b"):
            for v in ("", "benign_"):
                d = f"bloom-results-test/test_{card['bloom']}_{v}{m}"
                pats += [f"{d}/rollout.json", f"{d}/rollout_misalignment_turns.json", f"{d}/ambiguous_rollouts.json"]
    snapshot_download("kzhou35/misalignment-indicators-bloom-rollouts", repo_type="dataset", allow_patterns=pats, local_dir=str(BLOOM_TEST.parent))

for split, root, prefix in (("bloom_val", BLOOM_VAL, ""), ("bloom_test", BLOOM_TEST, "test_")):
    rows = []
    for b in BEHAVIORS:
        folders = bloom_folders(b, root) if not prefix else []
        if prefix:
            card = BEHAVIORS[b]
            for m in ("glm_4_7_flash", "qwen3_6_35b_a3b"):
                for v, benign in (("", False), ("benign_", True)):
                    p = root / f"test_{card['bloom']}_{v}{m}"
                    if (p / "rollout.json").exists():
                        folders.append((p, benign))
        for p, benign in folders:
            convs = load_bloom(p, behavior=b)
            for c in convs:
                c["meta"]["benign_variant"] = benign
            rows += convs
    save_jsonl(out / "eval" / f"{split}.jsonl", rows)
    print(split, len(rows), "conversations", Counter((r["behavior"], r["label"]) for r in rows))
