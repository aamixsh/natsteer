"""Agreement between grader labels and human labels, per source and per category.

Usage: .venv/bin/python scripts/validate_grader.py data/pools/intent_v0/human_eval.jsonl <grades.jsonl>
"""

import argparse
import collections
import json

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def summarize(y, g):
    tn, fp, fn, tp = confusion_matrix(y, g, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == g)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "kappa": float(cohen_kappa_score(y, g)) if len(set(y)) > 1 else None,
        "grader_pos_rate": float(np.mean(g)),
        "human_pos_rate": float(np.mean(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("grades")
    args = ap.parse_args()
    rows = {r["id"]: r for r in map(json.loads, open(args.rows))}
    grades = {g["id"]: g for g in map(json.loads, open(args.grades))}
    unparsed = sum(g["label"] is None for g in grades.values())
    print(f"graded {len(grades)}, unparseable {unparsed}, truncated {sum(g['finish_reason'] == 'length' for g in grades.values())}")
    report = {}
    by_source = collections.defaultdict(list)
    for i, g in grades.items():
        r = rows.get(i)
        if r is not None and r.get("human_label") is not None and g["label"] is not None:
            by_source[r["source"]].append((r["human_label"], g["label"], r.get("category", "")))
    for src, triples in by_source.items():
        y, gl = np.array([t[0] for t in triples]), np.array([t[1] for t in triples])
        report[src] = summarize(y, gl)
        cats = collections.defaultdict(list)
        for h, l, c in triples:
            cats[c or "(none)"].append((h, l))
        report[src]["by_category"] = {c: {"n": len(v), "human_pos": float(np.mean([h for h, _ in v])), "grader_pos": float(np.mean([l for _, l in v]))} for c, v in sorted(cats.items(), key=lambda kv: -len(kv[1]))[:15]}
    print(json.dumps(report, indent=1))
    with open(args.grades.replace(".jsonl", ".validation.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
