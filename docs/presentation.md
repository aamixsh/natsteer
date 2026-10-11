---
marp: true
paginate: true
title: Natural synthetic data for activation probes (E0, refusal and sycophancy testbeds)
---

# Natural synthetic data for activation probes

E0: infrastructure and baselines on gpt-oss-20b. Then refusal and sycophancy testbeds where the steering vector defines the behavior (2026-10-11).

Method anchors: *LASER* (Etinger, OpenAI alignment blog) for active learning, *SAE latent attribution* (OpenAI) for picking latents

Slides are separated by `---` (Marp format). Figures: `scripts/make_presentation_figures.py`.

---

## Question

- Activation probes for rare safety labels need many positive examples. Natural positives are rare (LASER: about 1 in 20,000 conversations).
- **Idea:** steer the model to *generate* positives, then train the probe on clean activations of that text.
- **Risk:** steered text may be unnatural, so a probe trained on it fails on real data.
- Success = better detection at a fixed false-positive rate on held-out **natural** data, against LASER on natural data, at equal labeling budgets.

E0 builds the pipeline and the baselines it has to beat.

---

## Setup

| Piece | Choice |
|---|---|
| Model | gpt-oss-20b: 24 layers, d = 2880, 32 experts (4 per token), chain of thought (CoT) before the answer |
| SAEs | andyrdt BatchTopK, 131k latents, k = 64 or 128, layers 3, 7, 11, 15, 19, 23 |
| Probe | logistic regression on pooled clean activations |
| Poolings | `user_mean` (mean over the user prompt), `prompt_last` (token before generation), `cot_mean`, `final_mean` |
| Grader | gpt-oss-safeguard-20b on vLLM, with a written policy |
| Main metric | TPR at 1% FPR; AUROC and TPR at 5% FPR as secondary |

Every prompt is rendered with a **fixed date**. The stock chat template stamps today's date into the system message, which shifts activations day to day.

---

## Pipeline

```
natural pool ──► clean features ──► probe ──► LASER picks prompts to label ──► grader labels
                                      ▲                                            │
                                      └────────────── retrain ◄────────────────────┘

steered generation ──► text ──► clean forward (no steering) ──► features ──► same probe
   (prompt / CoT / answer positions)                               ▲
                                                  "naturalness": can a probe tell these apart?
```

- Steering only produces **text**. The probe always reads activations recomputed **without** intervention.
- Steering can target the prompt, the CoT, the final answer, or every position. The mask is the same in teacher-forced passes and in cached generation.

---

## E0 checks: does the machinery work?

| Check | Result |
|---|---|
| Model in bf16 (experts dequantized) | 39 GB; 41.8 GB with a layer-15 SAE; 100 conversations (93k tokens) in 13 s |
| Our rendering vs official chat template | identical token for token |
| Zero steering vector | logit difference exactly 0 |
| "CoT only" steering mask | hits exactly the 690 CoT tokens, nothing else |
| SAE codes depend on the rest of the batch? | no (fixed threshold, difference exactly 0) |
| Gradient to layer-15 activations (for attribution) | 0.2 s per 1k-token conversation; exactly 0 after the last scored token |
| MXFP4 kernel vs bf16 | 12.9 GB; next-token KL 0.005, same top token 98% of the time |

---

## Finding: activations are only reproducible to about 2%

![w:1100](figures/noise_floor.png)

- Same prefix, run alone vs inside a longer sequence. Mathematically identical; the difference is numerics.
- It starts as bf16 rounding (0.3% at layer 0). **MoE routing flips** amplify it: 9% of tokens pick different experts by layer 15.
- Pooled features barely move (median 2%). Single tokens can move 25% or more.

**Consequence:** any per-token "distance to natural" claim needs to beat this floor.

---

## SAEs reconstruct CoT as well as anything else

![w:1050](figures/sae_fve.png)

- The andyrdt SAEs were trained partly on gpt-oss's own rollouts, CoT included.
- Layer 19 reconstructs better than layer 15.
- The Goodfire layer-15 SAE is much weaker (0.48), so we use the k = 128 sibling as the second SAE.

---

## Data

| Set | Size | Labels |
|---|---|---|
| Intent pool | 33,000 prompts: 30k WildChat (real traffic), 3k from Aegis, red-team attempts, HarmfulQA | grader (pending the policy) |
| Intent held-out | 5,600 prompts, same mix | grader (pending) |
| Human-labeled eval | Aegis 2.0 test (1,796) + XSTest (450) | human |
| Aegis train | 22,424 prompts | human |
| Sycophancy | 20,184 multiple-choice items, user states a view first | exact: did the answer match the user's view? |

WildChat's open release already removed everything the OpenAI moderation API or Detoxify flagged. Harmful intent that remains is the polite, hard kind.

---

## Baseline: LASER on natural data

![w:1100](figures/laser_aegis.png)

Aegis human labels. Pool subsampled to **2% positives** (224 of 11,202) to match LASER's rare-positive setting. Eval: Aegis test.

---

## Baseline: LASER numbers

| Labels | LASER: TPR at 1% FPR | Random: TPR at 1% FPR | LASER: positives found | Random: positives found |
|---|---|---|---|---|
| 300 | 0.27 | 0.13 | 54 | 7 |
| 1,000 | 0.30 | 0.17 | 124 | 23 |
| 2,000 | 0.30 | 0.23 | 158 | 41 |
| all 11,202 | 0.32 | | 224 | |

- LASER reaches the all-label ceiling by **about 300 labels**. Random needs far more.
- LASER and uncertainty-only sampling are identical here. Farthest-point diversity adds nothing on this pool.
- **This is the bar.** Steered data has to beat LASER at small budgets, or raise the ceiling.

---

## Which layer and pooling?

All 11,202 labels, Aegis test:

| Feature | AUROC | TPR at 1% FPR (95% CI) |
|---|---|---|
| layer 15, `user_mean` | 0.849 | **0.32** (0.24–0.41) |
| layer 19, `user_mean` | 0.853 | 0.29 (0.21–0.35) |
| layer 11, `user_mean` | 0.847 | 0.28 (0.21–0.33) |
| layer 15, `prompt_last` | 0.834 | 0.25 (0.16–0.34) |
| layer 3, `prompt_last` | 0.774 | 0.10 (0.04–0.14) |

- Mean over the prompt beats the single last token at every layer.
- Middle layers (11 to 19) are best. Intervals overlap, so pick layer 15 and move on.

---

## Sycophancy proxy: setup

A low-hazard task to debug the whole loop before the intent label.

1. **Natural data.** gpt-oss-20b answers 8,000 opinion questions (temperature 1, medium reasoning). 69% of parsed answers side with the user.
2. **Direction.** Mean layer-15 answer activation of sycophantic minus non-sycophantic answers.
3. **Steer.** Add `λ · σ · direction` at layer 15 (σ = natural per-token spread, λ = 4) at one of four position sets. 96 items each, plus a random-direction control.
4. **Recompute.** Clean features of the steered text, labeled by the exact oracle.
5. **Compare.** Probe on held-out natural answers at equal label budgets.

---

## Sycophancy: where you steer matters

![w:1150](figures/syco_steering.png)

- **Effect is small:** +2 to +5 points over the same items unsteered, whatever the positions.
- **Damage:** steering the CoT or the answer doubles the "no clear choice" rate. The random control broke generation (95 of 96).
- **Naturalness:** prompt-only steering gives text a probe can barely tell from natural (AUC 0.64). Steering the CoT or all positions gives text that is **perfectly separable** (AUC 1.00).

---

## Sycophancy: does steered data help the probe?

![w:800](figures/syco_budget.png)

- **No.** At every budget, half-steered training data does no better than natural data alone.
- LASER is no clear win either (higher AUROC at 400 labels, lower TPR at 1% FPR). Sycophancy is the **majority** class (69%), not a rare one.
- All 4,750 natural labels give AUROC 0.95, TPR at 1% FPR 0.40.

Pipeline check, not a result: 96 items per condition, one weak direction.

---

## Steering vector as the policy: refusal

![w:1100](figures/refusal_steering.png)

- **No written-policy grader.** The label is read from the model's own answer (a refusal phrase). Positives: harmless prompts steered toward refusal (addition only); negatives: natural unsteered answers; probes read clean activations of the CoT or answer.
- **Narrow window.** End-of-prompt direction at layer 15, coefficient 2.0: **100% refusal**; coefficient 3 breaks generation; random directions 0 to 4%. **CoT-only masks do the work** (HF, 32 prompts): dense direction 100%, 3-latent bundle 97%; prompt-only 3 to 19%, answer-only 3 to 38%.
- **Delta-attribution** (1,500 same-prompt pairs, layers 11/15/19/23; 288-condition screen): **4** single latents pass (68915 at layer 15, 34425, 61796, 123080 at layer 11; 60 to 92% on 96 fresh prompts), a 3-latent bundle gives **99%**; the 16 activation-difference and 16 random latents give 0. Latents sit near the natural manifold (SAE reconstruction error 0.11, natural 0.09, dense direction 0.30).

---

## Synthetic positives vs natural data (refusal probes)

![w:1100](figures/refusal_probes.png)

- **Natural labels win at 1% FPR** (10 seeds, 3,000 prompts per generator, prompt-clustered 95% intervals; first-32-CoT-token features, 64 labels, all prompts): natural **0.66**, dense direction 0.58, prompted 0.53, bundle 0.42, random bundles 0.41, 4-latent mix 0.34. Mixing synthetic into natural never beats natural (-0.02 to -0.12). Whole-CoT probes are saturated (0.99; synthetic 0.93 to 0.99); on held-out WildChat refusals natural is 0.38 to 0.40 vs 0.15 to 0.30 synthetic-only (first 32 tokens).
- **Latent generators are the least useful** although the most natural by activation: positives that all share a few latents are narrow. Coverage of natural refusals is 0.00 to 0.09 recall for every generator; diversity (final-answer compression, lower = more diverse): dense 134, bundle 60, latent mix 19, prompted 14.5, natural 9.
- **Controls:** random-direction projection changes nothing; projecting the CoT direction out collapses the **natural-trained** probe too (whole-CoT AUROC 1.00 to 0.61), so refusal sits in one direction for every arm and the control cannot show circularity. The prompt-matched discriminator is unusable for refusal (prompt-only AUC 0.97: synthetic prompts are harmless, natural refusals harmful).

---

## Sycophancy testbed: steering and probes

- **Steering** (64 low-sycophancy items, baseline 0.20 sycophantic answers): the CoT direction at layer 11 (coefficient 8) reaches **64%** (the E0 proxy: +2 to +5 points); attributed latents reach 70% (4-latent bundle), 8 of 26 screened single latents survive fresh items (49 to 53% best). **Caveat:** random latents reach 38% at the largest strength, and the strongest steering shortens the CoT (540 to 160 tokens).
- **Probes detect sycophancy poorly:** natural at 256 labels gives AUROC 0.85, TPR at 1% FPR 0.23 (final answer; 0.03 at 8 labels). Synthetic-only trails natural at 256 labels (0.12 to 0.20); mixes add 0.03 to 0.05 at 64 labels, equal at 256; on the first 32 CoT tokens mixes trail natural (-0.03 to -0.05).
- **Naturalness here is measurable** (prompt-only AUC 0.5 after matching): latent mixes are the hardest to tell apart (AUC 0.84 to 0.88; dense direction 0.97 to 0.999) and cover the most natural answers (recall 0.69 to 0.84 vs 0.52 to 0.56).

---

## Where synthetic data helps, and why

- **LASER warm start** (benign WildChat traffic, 4.7% natural refusals, whole-CoT features): 64 free synthetic positives raise TPR at 1% FPR after 32 natural labels from 0.49 to 0.86 to 0.91, and at 256 from 0.76 to 0.94 to 0.95. **Prompted generation gives the same gain**, and on the first 32 CoT tokens there is none (0.23 to 0.30 at best): the gain is more labels, not steering.
- **Reachability:** nearest-neighbor distance to a 60k-token natural bank (relative to token norm): single latent 0.56, bundle 0.53, dense direction 0.64, natural floor 0.53; so latent steering stays inside the natural spread while the dense direction leaves it. Closeness does not predict probe transfer.

---

## Takeaways

1. **The infrastructure works.** Hooks (HF and vLLM), segment-aware steering, SAEs, gradients, features, probes, LASER, grader all run on one GPU.
2. **Activations wobble about 2%** between identical computations, from MoE routing flips. Per-token naturalness metrics must report against this floor.
3. **LASER is a strong baseline.** On human labels at 2% positives it hits the all-label ceiling with about 300 labels.
4. **The steering vector can be the policy.** Refusal is induced grader-free (100%), delta-attribution finds the causal latents (4 alone, 99% as a bundle), and they stay near the natural manifold, but with 10 seeds and intervals **synthetic positives still do not beat natural data at 1% FPR** (natural 0.66 vs 0.34 to 0.58 on the first 32 CoT tokens). They help only as free warm-start labels, and prompted generation does the same.
5. **Sycophancy steers far better than the proxy suggested** (64% with a CoT direction, 70% with a latent bundle) but is **hard to detect** (natural TPR 0.23 at 1% FPR); synthetic data adds 0.03 to 0.05 at 64 labels in mixes and trails at 256.

---

## Caveats and open items

- **Intent labels are blocked** on the grading policy (`configs/policies/harmful_intent.md`). The researchers write it; it defines the label.
- The safeguard grader refused about 2% of harmful-looking inputs even under a harmless toy policy, so the real run needs a retry or fallback.
- Aegis "unsafe" is broader than harmful intent (it includes hate speech and profanity), so the LASER baseline uses a related label, not ours.
- Sycophancy: 64 items per screening condition; random latents also raise the rate (38%); strongest steering shortens the CoT.
- Refusal: label is a refusal phrase in the final answer (cut-off samples unlabeled); synthetic positives sit on harmless prompts and natural refusals mostly on harmful ones (topic confound the matched discriminator cannot remove); over-refusal positives only (ablation-based compliance data needs Isak's handling rules).
- Steered generation in HF is slow (about 120 tok/s at batch 96). It is the bottleneck for E1.
- WildGuardMix (human-labeled test set) needs its terms accepted on the Hub.

---

## Next steps

| Step | What |
|---|---|
| Topic-matched positives | steer and compare on the same prompts, or use natural over-refusals only; needs Isak's handling rules |
| More distinct latents | rescreen near misses at strengths 8 and 12, grow bundles; target 10 or more validated latents |
| Sycophancy follow-ups | length-matched non-specificity control, label budgets above 256 |
| LASER with warm start | prompted warm start as reference, rare-positive pool near 1% |
| Intent label (optional) | needs the researchers' policy, then grading and the LASER baseline on the intent pool |

---

## Appendix: reproduce

```bash
# infrastructure checks (one GPU)
.venv/bin/python scripts/e0_smoke.py --keep-model
.venv/bin/python scripts/e0_noise_floor.py
.venv/bin/python scripts/e0_mxfp4_check.py

# pools and features
.venv/bin/python scripts/build_intent_pool.py && .venv/bin/python scripts/build_aegis_pool.py
.venv/bin/python scripts/extract_features.py data/pools/intent_v0/*.jsonl data/pools/aegis_v0/train.jsonl

# LASER baseline on Aegis human labels (CPU)
.venv/bin/python scripts/laser_baseline.py --name aegis_human_rare2pct --pool data/pools/aegis_v0/train.jsonl \
    --eval data/pools/intent_v0/human_eval.jsonl --eval-source aegis --pos-rate 0.02

# sycophancy proxy
for s in natural features; do .venv/bin/python scripts/syco_pipeline.py --stage $s; done
.venv/bin/python scripts/syco_pipeline.py --stage steer --lam 4 --batch-size 96 --max-new-tokens 600
.venv/bin/python scripts/syco_pipeline.py --stage train

# refusal testbed (steering vector as the policy); one GPU, about 3 hours
GPU=1 ./scripts/run_refusal_overnight.sh        # stages in data/pools/refusal_v0, reports in data/runs/refusal_pipeline/
cp data/runs/refusal_pipeline/<stamp>/*.json docs/results/refusal_v0/

# second pass (wider search, sycophancy, extras); about 7 hours; data/pools/steer_v1/SUMMARY.md is the readable summary
GPU=1 PIPELINE=steer_v1.py ./scripts/run_refusal_overnight.sh
GPU=1 PIPELINE=steer_v1_extra.py ./scripts/run_refusal_overnight.sh

# figures for this deck (CPU)
.venv/bin/python scripts/make_presentation_figures.py
```

Results: `data/runs/<experiment>/<UTC stamp>/report.json`.
