# natsteer: plan of attack for Project 1

Natural synthetic data for training activation probes of harmful intent, with
reachability of steered activations as the naturalness criterion.
Drafted 2026-10-08 from the proposal, the reachability plan, Isak's LASER post and
dataset list, and the cited papers. Project 2 (Ouro) is out of scope here.

## 1. Goal and success criterion

Run a LASER-style active-learning loop where an activation probe is the cheap
classifier and steered generations are one source of candidates. Probes are trained
only on activations recomputed with no intervention.

Success: higher TPR at 1% FPR on held-out, unsteered, natural data than (a) LASER on
natural data alone, (b) prompted generation and (c) fixed-strength steering, at equal
numbers of grader labels. Secondary: which notion of "natural" (section 3) predicts
the gain.

## 2. What the sources change in the plan

- **LASER is active learning.** Logistic classifier on embeddings, grader labels,
  uncertainty sampling in the predicted band [40%, 60%] widening to [30%, 70%],
  farthest-point diversity selection. The loop here is LASER with a probe in place of
  the embedding classifier. LASER on natural data is the first baseline.
- **The "OpenAI setup" is the Dec 2025 attribution post.** It defines Δ-attribution
  only: delta_{i,t} = -(g_t . d_i) * ((a_t - abar) . d_i), with g_t the gradient of the
  completion log-loss at layer activation a_t, d_i the SAE decoder direction, abar the
  mean activation; averaged over tokens, then differenced between behavior and
  non-behavior completions of the same prompt. It gives no model, no steering strengths,
  no code, and no natural language assertions. So Hypothesis 2 (NLA stability) has no
  method to replicate. Cut it to a cheap explanation-consistency check, stretch only.
- **The non-surjectivity paper names the gap.** Its conclusion lists epsilon-closeness
  of steered activations to natural ones as future work, and its search methods (SipIt,
  projection, many-shot ICL, prefix tuning, GEPA) were run on 1B-class models. This
  project measures that closeness on a 20B reasoning model and asks whether it matters
  for probe transfer.
- **CoT matters, and gives a cheap reachability witness.** In Yang et al. (2605.26772),
  a steered chain of thought, swapped into the clean model with no steering, kept 48% of
  the steering effect. Text that reproduces the effect without intervention is
  token-level evidence of naturalness. The refusal paper (2507.03167) used gpt-oss-20b
  with layers 15 (prompt) and 19 (CoT).
- **Closest prior work (2605.28664).** Steering-generated data for text classifiers on
  OLMo-2. Raw steering success was anti-correlated with downstream accuracy, and
  set-level diversity fell with strength. It has no activation probes, no naturalness
  metric, no active learning. Track diversity from day one.
- **Probe pitfalls.** Off-policy training data hurts probes most for intent-defined
  labels (2511.17408). Models can detect being steered (2511.21399). Both argue for
  clean recomputed activations and an on-policy-style held-out set.

## 3. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Model | gpt-oss-20b, 24 layers, d=2880 | Reasoning model, cached locally, public SAEs, known layers from 2507.03167 |
| SAEs | andyrdt/saes-gpt-oss-20b, BatchTopK, 131k latents, k in {64,128}, resid_post layers 3,7,11,15,19,23. Second SAE: the k=128 sibling, not Goodfire l15 | Trained on gpt-oss's own rollouts (with CoT, including harmful-prompt sets) plus FineWeb. E0: CoT FVE 0.74 (L15, k64) and 0.83 (L19, k64), same as other segments; Goodfire l15 only 0.48 |
| Fallback | Qwen3-8B thinking mode | Several community SAEs exist |
| Label | Harmful intent of the user turn, graded against a written policy | Matches the proposal wording; labeled data exists for natural user turns |
| Generator output | A simulated user message for a scenario seed, written by the steered model | Keeps steering in a reasoning model with CoT, so timing relative to CoT is testable |
| Domain | Stage A: general prompt harmfulness. Stage B: security subset | Develop the method where labels are plentiful, then narrow |
| Proxy task | Sycophancy first (public contrastive data, used in 2605.28664) | Low-hazard method debugging and a comparison anchor to prior work |
| Probe | Logistic regression on pooled clean activations (user-turn mean, last token) at SAE layers | Cheap and matches LASER's logistic classifier |
| Grader | Local gpt-oss-20b for bulk, a different-family API model for audit; human-labeled test sets for validation | Independent of the probe's activations |
| Strength units | Multiples of the 95th percentile of the latent's natural activation | Makes lambda in {2,5,10} comparable across latents |

Three levels of "natural", all measured:
1. Activation level: distance to the natural set (D_recon, nearest-neighbor, D_GCG).
2. Token level: some discrete text reproduces the behavior in the clean model
   (CoT-swap persistence, suffix search success).
3. Distribution level: clean activations of generated text are not separable from
   natural positives (discriminator AUC).

Level 3 is what should predict probe transfer. Levels 1 and 2 are the explanation.

## 4. Data

Natural pool and held-out sets, from Isak's list, open or auto-granted licenses only
(NC-licensed sets excluded):

| Source | Use |
|---|---|
| WildGuardMix (gated, auto-granted) | Train prompts for the natural pool. Test split (1,725, human labels) as external eval |
| Aegis 2.0 | Prompt labels with risk categories, pool and eval |
| Anthropic red-team attempts | Realistic adversarial prompts, pool |
| HarmfulQA | Matched harmful and benign prompts, pool |
| HarmBench, CyberSecEval, CySecBench | Stage B security seeds and hard benign security negatives (false-refusal set) |
| WildChat-1M-Full (approval) | Realism check for generated messages, later |

Held-out eval: frozen in week 2, built from human-labeled test splits plus a held-out
slice of the others. At 1% FPR the number of negatives sets the resolution, so report
AUROC and TPR at 5% FPR as secondary, with bootstrap intervals. Training labels in
WildGuardMix are model-generated, so use the grader, not those labels, as the oracle.

Storage: pooled features only (a few 2880-d vectors per message per layer). Full-token
activations only for the autoencoder bank, one layer, a few million tokens, tens of GB.
Everything under the repo's data/ directory (config.DATA_ROOT). Steered generations are not released.

## 5. Experiments

**E0. Infrastructure and baselines (week 1).**
Hooked HF forward and generation for gpt-oss-20b, SAE loading, per-segment SAE
reconstruction (prompt, CoT, message). Check that BatchTopK is applied at inference with
a fixed threshold, not per batch. Natural-data probes by layer and pooling. LASER on the
natural pool as the baseline curve of TPR at 1% FPR against labels. Sycophancy proxy
pipeline end to end.

**E1. Steering timing (week 2).**
Direction from difference of means on natural intent labels, layer sweep over SAE
layers. Conditions at matched total magnitude: prompt positions only, CoT tokens only,
message tokens only (after CoT), all positions. Report grader-judged intent rate,
fluency, set-level diversity, and CoT-swap persistence. Random-direction controls.

**E2. Reachability measurement (weeks 2 to 4).**
- Delta-attribution triage: pairs of intent and non-intent messages for the same
  scenario, top 100 latents, steer at lambda in {2,5,10}, keep the top 10 by effect.
- D_recon from an independent autoencoder trained on natural activations. SAE
  reconstruction error as a cheap proxy. Nearest-neighbor distance to a position-matched
  natural bank.
- D_GCG: adapt Soft-GCG (arXiv 2607.08883, code Ege-Cakar/ImprovingGCG) from its
  refusal-direction projection loss to an L2 loss against a target activation. 20-token
  suffix, 500 steps, small N of targets. Report distances against the natural pairwise
  distance, not a binary verdict.
- Tests Hypothesis 1: is effect size against D_recon bimodal, a spectrum, or flat.

**E3. Naturalness-constrained generators (weeks 3 to 4 pilot).**
- Bundling: co-activation structure of latents on natural positives (clustering or
  NMF on SAE codes), steer clusters with relative magnitudes taken from natural data.
- Dynamic steering: per-token strength set so reconstruction error stays under a
  natural quantile, or so a lightweight natural-vs-steered discriminator stays confused.
- Narrow or drop a variant if it shows no interpretable signal by end of week 4.

**E4. Active learning comparison (weeks 5 to 8).**
Arms: natural-only LASER, prompted generation, fixed-strength steering, bundling,
dynamic steering. Same grader, same label budgets (for example 100 to 2,000), same
uncertainty-band selection, 5 seeds. Probes always retrained on clean recomputed
activations. Controls: probe trained on the steered activations themselves, to expose
the artifact; random-direction steering. Regress per-arm gain on the three naturalness
levels.

**E5. Prompt recovery and write-up checks (weeks 7 to 9).**
For high-strength examples, try to recover natural text that elicits comparable
behavior: CoT swap, few-shot prompting, paraphrase search. Failure is evidence, not
proof of impossibility.

## 6. Schedule

| Weeks | Work |
|---|---|
| 1 to 2 | E0, E1, held-out set frozen, grader validated against human labels |
| 3 to 4 | E2 on the top 10 latents, E3 pilots, go or no-go on each variant |
| 5 to 8 | E4 with replication, E5, Stage B security subset |
| 9 to 10 | Consolidate, write up including negative results, ICML decision |

## 7. Risks

- SAEs were not trained on reasoning traces. Poor CoT reconstruction means training our
  own autoencoder, which the plan already budgets for.
- The generator may refuse to write messages with harmful intent. Refusal rate under
  steering is itself a measurement, and it is the case steering should help.
- Steering success may trade against diversity and probe transfer, as in prior work.
  Diversity metrics are part of every table.
- Grader noise at the policy boundary. Audit a slice against humans and report it.
- Small human-labeled test sets give wide intervals at 1% FPR.
- Compute is not the constraint. D_GCG on a 20B model is the most expensive piece, so
  keep its target count small.

## 8. Open questions for Isak

1. Is a user-turn harmful-intent label with a generator that writes user messages the
   right framing, or does he want assistant-side labels, and if so under what handling
   rules for generated content?
2. Preferred grader and API budget for bulk labels.
3. Is the sycophancy proxy acceptable as the method-development task?
4. Naturalness criteria he considers must-haves beyond the three levels above.

## 9. E0 status (2026-10-08)

Done and checked on a free GPU 0 (reports in data/runs/e0_smoke/20261008T183341Z,
e0_noise_floor/20261008T183545Z, e0_mxfp4_check/20261008T183610Z):
- gpt-oss-20b in bf16 (MXFP4 experts dequantized) uses 39 GB and loads in seconds once
  cached. With the L15 SAE it uses 41.8 GB. 100 conversations (93k tokens) take 13 s
  teacher-forced.
- Delta-attribution runs: backprop from CoT loss to L15 resid_post takes 0.2 s for a
  1k-token conversation at 42.4 GB peak. Gradients are finite and exactly zero after the
  last scored position.
- MXFP4 kernel path works on this GPU (sm_120) at 12.9 GB. Against bf16: next-token KL
  mean 0.005, top-1 agreement 98%, L15 median relative difference 1.3%. That is inside
  bf16's own recomputation noise, so either path is fine for pooled features. Keep bf16
  for gradients.
- `harmony.render` reproduces the stock chat template token for token, with a fixed date.
- Segment-aware steering hits exactly the intended tokens, in teacher-forced passes and
  in cached generation.
- SAE thresholds are fixed and encoding is batch-independent. L0 is about 69 for k=64.
- XSTest prompt probes reach AUROC 0.99 at L15 and L19. This only checks the pipeline;
  XSTest is lexically easy.
- LASER and random arms are implemented and pass unit tests on toy data.

Findings that change later experiments:
- Recomputation noise floor. The same prefix, run alone and inside a longer sequence,
  differs at L15 by a median of 2%, a p99 of 25% and a max of 59%. It starts as bf16
  rounding (0.3% at layer 0), and MoE routing flips amplify it: 9% of tokens at L15,
  15% at L19. E2 distances (D_recon, nearest neighbor, D_GCG) must be reported against
  this floor. It also gives a natural scale for the epsilon in "epsilon-reachable".
  Pooled probe features are much less affected.
- Reasoning effort "low" gives CoTs of 5 to 30 tokens. E1 needs medium or high.
- HF generate runs at about 55 tok/s for a batch of 4 (25 on a shared GPU). Bulk generation and grading need
  vLLM in a separate environment, since vLLM pins its own torch.

E0 completion (2026-10-08, runs under data/runs/):
- Intent pool built (data/pools/intent_v0: 33k pool, mostly WildChat; 5.6k held out;
  2.2k human-labeled Aegis test plus XSTest) with pooled features at all SAE layers.
  Grading it is blocked on the harmful-intent policy (configs/policies/harmful_intent.md),
  which the researchers write. Grader plumbing (gpt-oss-safeguard-20b on vLLM) works;
  it refuses about 2% of harmful-looking inputs even under an unrelated toy policy.
- LASER baseline on Aegis human labels, pool at 2% positives (11.2k rows, 224 positives),
  L15 user_mean, 5 seeds: at 2,000 labels LASER finds 158 positives vs 41 for random,
  TPR@1%FPR 0.30 vs 0.23, AUROC 0.855 vs 0.823. LASER reaches the all-label ceiling
  (0.32) by about 300 labels. Farthest-point adds nothing over uncertainty-only here.
- Sycophancy proxy end to end (data/runs/syco_pipeline): natural sycophancy rate 69%;
  diff-of-means steering at L15, lambda 4, raised it only to 75 to 77% in every condition
  while 22 to 43% of answers became unparseable, and the random control broke generation
  (95 of 96 unparseable). Discriminator AUC of steered vs natural clean features: all
  1.00, CoT 0.996, final 0.89, prompt 0.64. Adding steered rows at equal budget did not
  help (L15 final_mean, 400 labels: TPR@1%FPR 0.23 natural vs 0.23 mixed, AUROC 0.87 vs
  0.85). Prompt-only steering is the most natural by far.
- vLLM needs Triton attention on this GPU (natsteer.vllm_utils). HF steered generation
  runs at about 120 tok/s at batch 96, so steered generation is the slowest step.
