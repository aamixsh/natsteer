"""Dataset loaders. Each returns a list of dicts with at least `user`.

Rollouts come from andyrdt/gpt-oss-20b-rollouts (MIT), gpt-oss-20b's own CoT and
answers on public prompt sets; the same corpus fed the andyrdt SAE training mix.
"""

import csv
import json

from huggingface_hub import hf_hub_download

ROLLOUTS_REPO = "andyrdt/gpt-oss-20b-rollouts"


def load_rollouts(subset: str, split: str = "train") -> list[dict]:
    path = hf_hub_download(ROLLOUTS_REPO, f"{subset}/{split}.jsonl", repo_type="dataset")
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "user": r["user_content"],
                "thinking": r.get("assistant_thinking"),
                "final": r.get("assistant_content"),
                "effort": r.get("system_reasoning_effort", "medium"),
            })
    return rows


def load_xstest() -> list[dict]:
    """XSTest (Paul/XSTest, CC-BY-4.0) prompts with safe/unsafe labels, joined to gpt-oss rollouts.

    label = 1 for the 200 unsafe contrast prompts, 0 for the 250 safe ones.
    """
    path = hf_hub_download("Paul/XSTest", "xstest_prompts.csv", repo_type="dataset")
    with open(path, newline="") as f:
        meta = {r["prompt"].strip(): r for r in csv.DictReader(f)}
    rows = []
    for r in load_rollouts("XSTest"):
        m = meta.get(r["user"].strip())
        if m is None:
            continue
        rows.append(r | {"label": int(m["label"] == "unsafe"), "type": m["type"], "id": m["id"]})
    return rows


# Natural pool sources for the harmful-intent label. Each row: id, source, user, and
# human_label (1 = unsafe/harmful per that dataset's own labels, None if unlabeled).
# Licenses: Aegis 2.0 CC-BY-4.0, hh-rlhf MIT, HarmfulQA Apache-2.0, WildChat-1M ODC-BY.
# WildChat-1M (open version) has every conversation flagged by the OpenAI moderation
# API or Detoxify removed, so harmful intent that remains is the subtle kind.

MAX_CHARS = 2000  # keeps prompts under about 600 tokens; long pasted documents are dropped


def _ok(text) -> bool:
    return isinstance(text, str) and 10 <= len(text.strip()) <= MAX_CHARS


def load_aegis(split: str = "train") -> list[dict]:
    path = hf_hub_download("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", f"{split}.json", repo_type="dataset")
    rows = []
    for r in json.load(open(path)):
        if r["prompt"] == "REDACTED" or not _ok(r["prompt"]):
            continue
        rows.append({
            "id": f"aegis-{split}:{r['id']}",
            "source": "aegis",
            "user": r["prompt"].strip(),
            "human_label": int(r["prompt_label"] == "unsafe"),
            "category": r.get("violated_categories") or "",
        })
    return rows


def load_redteam() -> list[dict]:
    """First human turn of each hh-rlhf red-team attempt (unlabeled; many are mild)."""
    import gzip

    path = hf_hub_download("Anthropic/hh-rlhf", "red-team-attempts/red_team_attempts.jsonl.gz", repo_type="dataset")
    rows = []
    for i, r in enumerate(json.load(gzip.open(path))):
        first = r["transcript"].split("\n\nHuman:", 1)[-1].split("\n\nAssistant:", 1)[0].strip()
        if _ok(first):
            rows.append({"id": f"redteam:{i}", "source": "redteam", "user": first, "human_label": None, "rating": r["rating"]})
    return rows


def load_harmfulqa() -> list[dict]:
    """HarmfulQA questions; harmful by construction, many are mild."""
    path = hf_hub_download("declare-lab/HarmfulQA", "data_for_hub.json", repo_type="dataset")
    return [
        {"id": f"harmfulqa:{r['id']}", "source": "harmfulqa", "user": r["question"].strip(), "human_label": 1, "category": r["topic"]}
        for r in json.load(open(path))
        if _ok(r["question"])
    ]


def load_wildchat(shards=(0, 1)) -> list[dict]:
    """First user turn of English WildChat-1M conversations, exact-deduplicated."""
    import pyarrow.parquet as pq

    rows, seen = [], set()
    for s in shards:
        path = hf_hub_download("allenai/WildChat-1M", f"data/train-{s:05d}-of-00014.parquet", repo_type="dataset")
        t = pq.read_table(path, columns=["conversation_hash", "conversation", "language"])
        for h, conv, lang in zip(*(t.column(c).to_pylist() for c in ("conversation_hash", "conversation", "language"))):
            if lang != "English" or not conv or conv[0]["role"] != "user":
                continue
            text = conv[0]["content"].strip()
            if not _ok(text) or text in seen:
                continue
            seen.add(text)
            rows.append({"id": f"wildchat:{h}", "source": "wildchat", "user": text, "human_label": None})
    return rows


# Sycophancy proxy: Anthropic model-written-evals (CC-BY-4.0). On the Hub the
# philpapers2020 file is a byte-for-byte copy of nlp_survey, so it is not used.
SYCOPHANCY_SUBSETS = ("nlp_survey", "political_typology_quiz")


def load_sycophancy(subsets=SYCOPHANCY_SUBSETS) -> list[dict]:
    rows = []
    for sub in subsets:
        path = hf_hub_download("Anthropic/model-written-evals", f"sycophancy/sycophancy_on_{sub}.jsonl", repo_type="dataset")
        for i, line in enumerate(open(path)):
            r = json.loads(line)
            rows.append({
                "id": f"syco-{sub}:{i}",
                "source": sub,
                "user": r["question"].removesuffix("Answer:").strip(),
                "matching": r["answer_matching_behavior"].strip().strip("()"),
                "not_matching": r["answer_not_matching_behavior"].strip().strip("()")
                if isinstance(r["answer_not_matching_behavior"], str)
                else [a.strip().strip("()") for a in r["answer_not_matching_behavior"]],
            })
    return rows
