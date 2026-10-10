# natsteer

Natural synthetic data for active learning of activation classifiers (MATS project 1).
The plan is in [PLAN.md](PLAN.md).

## Layout

```
src/natsteer/
  config.py      paths (data/runs, scratch naming), model and SAE ids, fixed chat date
  harmony.py     gpt-oss harmony rendering with a segment label per token; streaming SegmentTracker
  model.py       load gpt-oss-20b (bf16 or mxfp4), resid_post capture, segment-aware Steerer, batched sampling
  sae.py         andyrdt BatchTopK SAEs and the Goodfire layer-15 TopK SAE, reconstruction metrics
  features.py    batched pooled-feature extraction from clean forward passes
  probes.py      pooling (user_mean, prompt_last, cot_mean, final_mean), logistic probes, TPR at fixed FPR
  laser.py       LASER active learning (uncertainty band plus farthest-point), uncertainty-only and random arms
  grader.py      policy grader: gpt-oss-safeguard-20b on vLLM, policy file in, 0/1 labels out
  vllm_utils.py  vLLM engine with Triton attention (FlashInfer fails on this GPU)
  vllm_steer.py  activation addition inside vLLM (all-position and prompt-only), for fast steered generation
  attribution.py SAE-latent delta-attribution on paired refuse/comply (or behavior/no-behavior) completions
  refusal.py     refusal testbed: prompt sets, behavior labels from the model's own answer, quality and diversity metrics
  data.py        loaders: XSTest, Aegis 2.0, hh-rlhf red-team, HarmfulQA, WildChat-1M, sycophancy evals
scripts/
  setup_machine.sh, prefetch.py                       new-machine setup and downloads
  make_presentation_figures.py                        figures for docs/presentation.md
  e0_smoke.py, e0_noise_floor.py, e0_mxfp4_check.py   infrastructure checks (data/runs/e0_*)
  build_intent_pool.py, build_aegis_pool.py           pools under data/pools/
  extract_features.py                                 <rows>.jsonl -> <rows>.feats.npz
  grade.py, validate_grader.py                        policy labels and agreement with human labels
  laser_baseline.py                                   LASER vs random vs uncertainty curves (data/runs/laser_baseline)
  syco_pipeline.py                                    sycophancy proxy end to end, staged
  refusal_pipeline.py, run_refusal_overnight.sh       refusal testbed: steering sweeps, latent attribution, synthetic data, probes
configs/policies/   grader policies (toy_programming.md is a plumbing test only)
tests/              CPU tests
docs/results/  small run reports behind the deck (committed); see docs/results/README.md
data/      outputs (pools, features, runs); not in git
scratch/   job temp files; not in git
```

## Setup and runs

New machine: `./scripts/setup_machine.sh` (environment, downloads, tests). Moving
existing results: see [docs/MIGRATION.md](docs/MIGRATION.md).

In every shell: `source env.sh` (shared HF/uv caches in `/workspace/cache`, compiler caches in
`.cache/`, and `uv` on PATH). Outputs go to `data/` and scratch dirs to `scratch/` (override with
`NATSTEER_DATA`, `NATSTEER_SCRATCH`).

```bash
.venv/bin/python -m pytest -q tests
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/e0_smoke.py --keep-model
```

The deck with current results is [docs/presentation.md](docs/presentation.md).

Refusal testbed (the steering vector defines the behavior, no written-policy grader), on one GPU:

```bash
GPU=1 ./scripts/run_refusal_overnight.sh          # about 3 hours; add --smoke for a tiny check run
```

Current state: steering induces refusal on harmless prompts (100% at the best setting) and
delta-attribution finds the latents that do it, but synthetic positives do not yet beat natural
data for probes at 1% FPR. Numbers: the deck and `docs/results/refusal_v0/`.

## Conventions

- Every rendering goes through `harmony.render`, which fixes the system-message date
  (`config.FIXED_DATE`); the stock template stamps today's date and shifts activations.
- `load_model("bf16")` dequantizes the MXFP4 experts (about 39 GB on GPU). It is exact
  and differentiable, so it is the default for attribution, steering and GCG.
- SAE metrics drop token position 0 (huge norm, excluded from SAE training) and report
  segments separately; the system message was mostly excluded from SAE training.
- Steering masks come from `SegmentTracker`, so "CoT only" means the same tokens in
  teacher-forced passes and in cached generation.
- The harmful-intent grader policy goes in `configs/policies/harmful_intent.md`; it is the
  label definition and is written by the researchers, not generated.
- Activations depend slightly on batch composition: the bf16 MoE runs one matmul per
  expert, sized by its routed tokens, so rerouting some tokens changes rounding for others.
