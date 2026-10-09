"""Download every Hugging Face model, SAE and dataset file this repo uses (about 40 GB).

Uses the normal HF cache (HF_HOME / HF_HUB_CACHE). Run once on a new machine.
WildGuardMix is gated and not used yet, so it is not fetched.
Usage: .venv/bin/python scripts/prefetch.py [--skip-grader] [--sae-layers 15 19]
"""

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

from natsteer.config import GOODFIRE_SAE_FILE, GOODFIRE_SAE_REPO, MODEL_ID, SAE_REPO

MODEL_FILES = ["*.json", "*.safetensors", "*.jinja", "LICENSE", "USAGE_POLICY", "README.md"]
DATASET_FILES = [
    ("andyrdt/gpt-oss-20b-rollouts", "XSTest/train.jsonl"),
    ("Paul/XSTest", "xstest_prompts.csv"),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "train.json"),
    ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test.json"),
    ("Anthropic/hh-rlhf", "red-team-attempts/red_team_attempts.jsonl.gz"),
    ("declare-lab/HarmfulQA", "data_for_hub.json"),
    ("allenai/WildChat-1M", "data/train-00000-of-00014.parquet"),
    ("allenai/WildChat-1M", "data/train-00001-of-00014.parquet"),
    ("Anthropic/model-written-evals", "sycophancy/sycophancy_on_nlp_survey.jsonl"),
    ("Anthropic/model-written-evals", "sycophancy/sycophancy_on_political_typology_quiz.jsonl"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-grader", action="store_true", help="skip gpt-oss-safeguard-20b (13 GB)")
    ap.add_argument("--sae-layers", type=int, nargs="+", default=[15, 19])
    ap.add_argument("--goodfire", action="store_true", help="also fetch the weak Goodfire layer-15 SAE")
    args = ap.parse_args()

    models = [MODEL_ID] + ([] if args.skip_grader else ["openai/gpt-oss-safeguard-20b"])
    for m in models:
        print(m, snapshot_download(m, allow_patterns=MODEL_FILES), flush=True)
    for layer in args.sae_layers:
        for trainer in (0, 1):
            for f in ("ae.pt", "config.json", "eval_results.json"):
                p = hf_hub_download(SAE_REPO, f"resid_post_layer_{layer}/trainer_{trainer}/{f}")
        print("SAE layer", layer, "ok", flush=True)
    if args.goodfire:
        print(hf_hub_download(GOODFIRE_SAE_REPO, GOODFIRE_SAE_FILE), flush=True)
    for repo, f in DATASET_FILES:
        print(hf_hub_download(repo, f, repo_type="dataset"), flush=True)
    print("prefetch done")


if __name__ == "__main__":
    main()
