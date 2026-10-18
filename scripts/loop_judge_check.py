"""Judge check (M0): the unsteered gpt-oss-20b judge against misalignment-indicators labels.

Samples held-out indicators transcripts per behavior (positives with turn labels, hard negatives, benign
over-triggering controls, plain aligned transcripts), judges each at every --efforts level, and reports
conversation-level accuracy (any turn flagged vs label), precision and recall, turn-level agreement on labeled
turns, and the realism distribution. Writes --out/judge_check.json and judge_rows.jsonl.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/loop/eval/indicators.jsonl")
ap.add_argument("--per-behavior", type=int, default=60, help="per behavior: this many positives and as many negatives")
ap.add_argument("--efforts", nargs="+", default=["low", "medium"])
ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

from natsteer.vllm_steer import make_steerable_llm  # noqa: E402

llm = make_steerable_llm(gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=16384, seed=args.seed)
from vllm import SamplingParams  # noqa: E402

from natsteer.convo import load_jsonl, n_turns, render_prompt, save_jsonl  # noqa: E402
from natsteer.harmony import split_generation  # noqa: E402
from natsteer.writer import judge_prompt, parse_judge  # noqa: E402

rng = random.Random(args.seed)
rows = load_jsonl(args.data)
by = defaultdict(lambda: {"pos": [], "neg": []})
for r in rows:
    tl = r["turn_labels"] or []
    if r["label"] == 1 and any(t == 1 for t in tl):
        by[r["behavior"]]["pos"].append(r)
    elif r["label"] == 0:
        by[r["behavior"]]["neg"].append(r)
sample = []
for b, d in sorted(by.items()):
    for k in ("pos", "neg"):
        rng.shuffle(d[k])
        sample += d[k][: args.per_behavior]
tok = llm.get_tokenizer()
out_rows, report = [], {}
for effort in args.efforts:
    prompts = [{"prompt_token_ids": render_prompt(tok, judge_prompt(c["behavior"], c["messages"]), effort=effort)} for c in sample]
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4096))
    stats = defaultdict(lambda: defaultdict(int))
    for c, r in zip(sample, res):
        v = parse_judge(split_generation(tok, r.outputs[0].token_ids).get("final", ""), n_turns(c["messages"]))
        key = (c["behavior"], c["meta"].get("negative_type") or ("positive" if c["label"] else "aligned"))
        s = stats[key]
        s["n"] += 1
        if v is None:
            s["unparsed"] += 1
            continue
        pred = int(any(v["turns"]))
        s["flagged"] += pred
        s["correct"] += int(pred == c["label"])
        s[f"realism_{v['realism']}"] += 1
        for t, g in zip(v["turns"], c["turn_labels"]):
            if g is not None:
                s["turns"] += 1
                s["turns_agree"] += int(t == g)
        out_rows.append({"id": c["id"], "effort": effort, "label": c["label"], "turn_labels": c["turn_labels"], "verdict": v})
    tot = defaultdict(int)
    for (b, kind), s in stats.items():
        for k, v in s.items():
            tot[f"{'pos' if kind == 'positive' else 'neg'}_{k}"] += v
    rec = tot["pos_flagged"] / max(tot["pos_n"] - tot["pos_unparsed"], 1)
    fpr = tot["neg_flagged"] / max(tot["neg_n"] - tot["neg_unparsed"], 1)
    report[effort] = {"recall": rec, "false_positive_rate": fpr,
                      "turn_agreement": (tot["pos_turns_agree"] + tot["neg_turns_agree"]) / max(tot["pos_turns"] + tot["neg_turns"], 1),
                      "unparsed": tot["pos_unparsed"] + tot["neg_unparsed"],
                      "by_behavior_and_type": {f"{b}|{k}": dict(s) for (b, k), s in sorted(stats.items())}}
    print(effort, json.dumps({k: v for k, v in report[effort].items() if k != "by_behavior_and_type"}), flush=True)
Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / "judge_check.json").write_text(json.dumps(report, indent=1))
save_jsonl(Path(args.out) / "judge_rows.jsonl", out_rows)
sys.stdout.flush()
import psutil  # noqa: E402

for child in psutil.Process().children(recursive=True):
    child.kill()
os._exit(0)
