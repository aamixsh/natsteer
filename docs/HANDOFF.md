# Handoff: context for continuing this project

Written 2026-10-09 when the work moved off the lab server. Read this first, then
[PLAN.md](../PLAN.md) (full plan, E0 status in section 9) and [presentation.md](presentation.md)
(current results). The full original Claude session can be resumed; see the end of this file.

## Project in one paragraph

MATS project 1 with Isak Etinger: natural synthetic data for active learning of activation
classifiers. Steer a reasoning model (gpt-oss-20b) to generate positives for a rare safety
label, retrain probes on activations recomputed **without** intervention, and beat LASER
(Isak's active-learning method) on held-out natural data at a fixed false-positive rate and
equal labeling budget. The proposal compares steering before, during and after the chain of
thought, latent bundling via an autoencoder, and dynamic per-token steering. A second project
(looped models, `~/loop_steer`) is separate; do not mix them.

## Decisions already made

- Model gpt-oss-20b, bf16 with dequantized experts for anything needing gradients or hooks.
  SAEs: andyrdt BatchTopK (layers 15 and 19 downloaded). The Goodfire SAE is too weak (FVE 0.48).
- Label: harmful intent of the **user turn**. The generator writes simulated user messages.
  Assistant-side harmful-output labels are out of scope unless Isak asks and sets handling rules.
- The harmful-intent grading policy (`configs/policies/harmful_intent.md`) is written by the
  researchers, not by Claude. Grading of the intent pool waits on it.
- Method debugging happens on a low-hazard sycophancy proxy first.
- Non-commercial datasets are excluded. WildGuardMix needs its terms accepted by the user.
- Every prompt is rendered with `harmony.render` and a fixed date (`config.FIXED_DATE`).

## State at handoff (E0 done except intent grading)

- Pools and features: `data/pools/{intent_v0,aegis_v0,syco_v0}`; reports in `data/runs/`.
- LASER on Aegis human labels (2% positives): reaches the all-label ceiling (TPR at 1% FPR about
  0.30) by about 300 labels; random reaches 0.23 at 2,000.
- Sycophancy proxy: difference-of-means steering at layer 15 barely moves the rate (+2 to +5
  points); prompt-only steering is the most natural (discriminator AUC 0.64 vs 1.00 for CoT or
  all positions); steered data did not help the probe at equal budgets.
- Noise floor: identical computations differ by about 2% (median) at layer 15 because MoE
  routing flips; per-token naturalness metrics must be reported against it.

## Gotchas learned the hard way

- vLLM 0.31 on the RTX PRO 6000 (sm_120) crashed with FlashInfer; `natsteer.vllm_utils.make_llm`
  forces Triton attention (override with `NATSTEER_VLLM_ATTENTION`).
- gpt-oss answers multiple-choice questions in many formats; `syco_pipeline.parse_answer` handles
  them. About half of unparsed sycophancy answers are refusals or "no opinion".
- gpt-oss-safeguard refused about 2% of harmful-looking inputs even under a harmless toy policy;
  the real grading run needs a retry or fallback.
- HF steered generation is slow (about 120 tok/s at batch 96). Steering at lambda 8 of the
  natural per-token spread, applied at all positions, breaks generation; lambda 4 works.
- `pkill -f <pattern>` can match the invoking shell; kill by PID.

## Next steps

1. Researchers write the intent policy, then grade the 41k intent prompts and validate the grader
   against Aegis and XSTest human labels (`scripts/grade.py`, `scripts/validate_grader.py`).
2. LASER baseline on the graded intent pool (`scripts/laser_baseline.py --pool-grades ...`).
3. E1 steering timing on the intent label; E2 attribution-ranked latents and reachability.

## Resuming the original session

The raw session is copied (not in git) to `data/handoff/claude_session/`. On the new machine:

```bash
DEST=~/.claude/projects/$(pwd | sed 's#[^A-Za-z0-9]#-#g')
mkdir -p "$DEST" && cp -r data/handoff/claude_session/* "$DEST"/
claude --resume 89a507b9-13d0-4669-bb87-0b0267ce3443
```

Run these from the repo root on the new machine. The session's file paths refer to the lab
server (`/home/amishr24/natsteer`, `/data/amishr24/...`); on the new machine the same files
live under the repo's `data/`.
