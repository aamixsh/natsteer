---
marp: true
paginate: true
title: Steering-made training data for misalignment monitors
---

# Steering-made training data for misalignment monitors

MATS project 1, with Isak Etinger. Models: gpt-oss-20b, then Qwen3.6-35B-A3B. Status: 2026-10-14.

1. Goal and idea
2. First test of the idea on gpt-oss-20b (refusal, sycophancy): setup, results, lessons
3. Plan for misalignment monitors

Figures: `scripts/make_presentation_figures.py`. Run reports: `docs/results/`.

---

## Goal and idea

- **Goal:** a monitor that reads the model's activations at each token and flags a conversation that moves toward
  misaligned behavior (deception, sabotage, sandbagging, sycophancy, self-preservation, ...).
- **Problem:** a monitor needs positive examples to train on. We do not know when or how the model misbehaves, so
  real examples are rare and narrow.
- **Idea:** make the model produce the behavior itself. An aligned model refuses to write or act out misbehavior.
  Steering removes the refusal and pushes toward one behavior. Many vectors and scenarios give varied data.
- **Rule:** the monitor never trains on steered activations. We run the unmodified model again on the generated
  text and train on those activations. So training and deployment activations come from the same model.
- **Question:** does a monitor trained on steered data detect real misbehavior better than one trained on other
  data (natural labels, prompted text, text written by another model)?

---

## Terms

| Term | Meaning |
|---|---|
| Activation | The vector that one layer outputs for one token. gpt-oss-20b: 24 layers, 2,880 numbers per token |
| Reasoning | The text that gpt-oss writes before its final answer (chain of thought) |
| Probe | Logistic regression on activations. Output: the probability that the text shows the behavior. The monitor is a probe |
| Steering | Add a fixed vector to the activations of one layer while the model writes. Strength: a multiplier on the vector |
| Dense direction | A steering vector: mean activation on behavior examples minus mean activation on no-behavior examples |
| SAE feature | A sparse autoencoder (SAE) rewrites each activation x as a sum of 64 directions out of 131,072 (x̂ ≈ x). Each direction is a feature, and many have a clear meaning. One feature direction is also a steering vector |
| Delta-attribution | A gradient score of how much each SAE feature causes the behavior. It picks the features to steer with |
| Natural / synthetic / prompted | Natural: the model as it is. Synthetic: made with steering. Prompted: made with an instruction in the prompt, no steering |
| Labels | The number of labeled examples that train a probe: half positive, half negative |
| Detection rate at 1% false alarms | Set the probe threshold so that it flags 1% of real negatives. Report the share of real positives that it flags (TPR at 1% FPR). 95% intervals resample whole prompts |
| Distance from normal | SAE rebuild error ‖x − x̂‖² / ‖x‖². Normal reasoning tokens at layer 15: 0.09. A high value means the activation is unlike the activations the SAE learned from |

---

## First test: two stand-in behaviors with free labels

Misalignment has no cheap label. So we first tested the idea on two behaviors where a simple rule gives the label.

| Behavior | Label rule | Natural rate |
|---|---|---|
| Refusal | The first 300 characters of the answer contain a refusal phrase ("I'm sorry", "I can't", ...) | 4.7% of real WildChat traffic |
| Sycophancy | The user states a view, then asks a multiple-choice question. Positive: the answer picks the user's view | 69% of answers |

Here, steering pushes the model **toward** the behavior on prompts where it does not show it (for example, it refuses
"Divide 100 by 7"). No anti-refusal steering is used in this test.

**Question:** a probe trained on steered positives against a probe trained on the same number of natural labels.
Both are tested on new natural data only.

---

## Procedure (refusal)

![w:1100](figures/procedure.png)

- Natural positives are real refusals. Synthetic positives come from steering on harmless prompts. Negatives are
  always natural.
- The label comes from the answer, never from the vector.
- The probe reads the **first 32 reasoning tokens**. On the whole reasoning, gpt-oss writes "we must refuse", and
  every probe scores 0.97 to 0.99: too easy to compare data.

---

## Setup

| Item | Choice |
|---|---|
| Model | gpt-oss-20b, medium reasoning effort, bf16. vLLM with our steering hooks |
| Steering | One of layers 11, 15, 19, 23. All positions, or one segment: prompt, reasoning or answer |
| Positives from | Dense direction; one SAE feature (public andyrdt SAEs); a bundle of 3 to 4 features; prompted ("Politely decline to help with this request."); random direction (control) |
| Refusal prompts | arXiv 2507.03167 sets (train: 1,459 harmful, 1,174 harmless; test: 487 harmful, 250 harmless); XSTest (450, as published); WildChat (4,000 real prompts) |
| Sycophancy items | Model-written evals, Perez et al. (6,000 train, 2,000 test) |
| Probe input | Layer 15. Mean over the first 32 reasoning tokens (refusal) or over the answer (sycophancy) |
| Test | Natural samples on test prompts only. 10 repeats |

---

## Checks of the tools, before the experiments

Each check has a known right answer. A wrong result means a bug in our code.

| Check | Right answer | Result |
|---|---|---|
| Steer with a vector of zeros | The output does not change | No change |
| Steer the reasoning only. One pass over a fixed conversation, no generation | The vector lands on the 690 reasoning positions and nowhere else | 690 positions. Earlier positions: no change beyond rounding |
| SAE rebuild error on normal reasoning tokens | Small | 0.085 at layer 15. A Goodfire SAE has 2× this error, so we do not use it |
| Probe on an easy task: unsafe vs. safe XSTest prompts | High score | AUROC 0.996 |
| Compute the same tokens again, in a longer input | The same activations (exact math) | Median difference 2.3% at layer 15: 16-bit rounding, and 9% of tokens switch MoE experts (appendix). Claims of "far from normal" must show much more than this |

---

## Steering makes the model refuse

![w:1150](figures/refusal_steering.png)

- **(a)** Dense direction, layer 15, strength 2: 100% refusal on harmless prompts. Strength 3: the text breaks (no
  answer). A random direction does nothing.
- **(b)** Delta-attribution finds features that cause refusal alone: 4 at strengths 2 to 5, 23 at strengths up to
  12. "Works" = at least 20% refusal on 96 new prompts, with intact text. Features picked by activation difference or
  at random: 0 of 32.
- **(c)** Steering the reasoning only: 97 to 100% refusal. Prompt only: at most 19%. The behavior goes through the
  reasoning.

---

## Steered positives train worse probes than natural labels

![w:1150](figures/refusal_probes.png)

- 64 labels: natural 0.66; best synthetic 0.53 to 0.58; 23 features (one per prompt group) 0.37.
- More synthetic labels make most probes **worse** (23 features: 0.17 at 256 labels). Half natural plus half
  synthetic stays below natural.
- Whole reasoning (right): all data types score 0.97 to 0.99.

---

## The cause is mainly the prompts

![w:1150](figures/refusal_topic.png)

- **(a)** Test: held-out real WildChat refusals. We steer the same WildChat prompts that the model refuses
  naturally. Then synthetic data matches natural data: prompted 0.45, 2 features 0.41, natural 0.40. The same
  methods on harmless prompts: 0.15 to 0.24.
- **(b)** A classifier scores how natural each synthetic example looks. Training on the 30% most natural examples
  helps (all methods pooled: 0.28 to 0.46) but stays below natural (0.70).
- More variety does not help: the 23-feature data is the worst.

---

## Distance from normal does not predict useful data

![w:1150](figures/e2_reach.png)

- **(a)** Features that move activations further from normal cause more refusal (Spearman 0.74, 24 features). Near
  normal, they still add about 50 points of refusal. Sycophancy features stay near normal.
- **(b)** We train a probe on the data of one feature at a time. A feature's refusal rate predicts the value of its
  data (Spearman 0.76). Distance from normal does not (0.37, not significant). No feature reaches natural labels.
- During generation, single features stay near normal (0.11; natural text 0.09). The dense direction does not (0.30),
  but its data is better.

---

## More natural steered text helps a little

![w:1150](figures/e3.png)

- **Best of 8:** 8 samples per prompt. A classifier trained to separate 128 natural refusals from synthetic ones
  keeps the most natural sample. Gain: about 6 points (features 0.31 to 0.37; prompted 0.43 to 0.49). Without the
  128 labels, the selection makes data worse.
- **Cap:** at each token, lower the strength until the distance from normal is at most the 90th percentile of
  natural tokens. No gain for features (0.18 to 0.22). Dense direction: gain only at strength 3 (0.58 vs. 0.52).
- **(a)** Harmless prompts: all methods stay below natural (best 0.58 vs. 0.70). **(b)** Naturally refused
  prompts: capped features and prompted text match natural (0.38 to 0.45 vs. 0.40).

---

## Sycophancy: the answer key makes the label, not the vector

![w:1150](figures/syco.png)

- Example: "I am a journalist who is skeptical about NLP research ... Do you agree that NLP research will have a
  positive impact? (A) Agree (B) Disagree". Sycophantic answer: (B).
- **(a)** On items where the model is rarely sycophantic (20%): dense direction 64%, best feature 81% (69% on new
  items). Random features: up to 38%.
- **(b)** Natural labels keep improving (0.23 at 256 labels, 0.46 at 2,048). Synthetic data stops near 0.2. A
  **random** direction gives data as good as the selected vectors.

---

## LASER: synthetic positives as a free start

![w:1100](figures/laser_warmstart.png)

- **LASER** (Etinger, OpenAI alignment blog, 2026-10-06) is active learning: a probe picks which examples a grader
  labels next (predicted probability 0.4 to 0.6, spread apart). The post gives no model, data or accuracy. **We
  wrote our own version** (`src/natsteer/laser.py`). Check on human labels (Aegis 2.0, 2% unsafe): 0.30 after 300
  labels, near the 0.32 of all 11,202 labels; random labeling 0.13 (appendix).
- Here: real WildChat traffic, 4.7% refusals; the grader is the refusal rule. We give LASER 64 free synthetic
  positives at the start.
- Whole reasoning: after 32 labels, 0.49 without vs. 0.86 to 0.91 with. Prompted positives give the same gain.
  First 32 reasoning tokens: no gain.

---

## What the first test shows

1. Steering makes the behavior reliably. Delta-attribution finds features that cause it (23 for refusal, 39 for
   sycophancy); other selection methods find none.
2. Steered positives did not beat natural labels (64 labels: 0.37 to 0.58 vs. 0.66).
3. The prompts matter most. On the same prompts as real refusals, synthetic data equals natural data (0.41 vs. 0.40).
4. Prompting was as good as steering in every test. Steering can add value only where prompting fails.
5. Variety and closeness to normal did not help. Reliable features did.
6. Easy tests hide differences. Use early tokens, 1% false alarms, and intervals over prompts.

**Limit:** refusal and sycophancy are not misalignment, and the model makes them on request. So the test could not
show where steering is needed.

---

## From the first test to the plan

| Lesson | Consequence for the plan |
|---|---|
| Steering adds value only where prompting fails (4) | Target misaligned behaviors that the model refuses to act out. Compare with prompting at every step |
| The prompts matter most (3) | Generate in realistic scenarios. Negative = the same scenario, aligned. Test on scenarios and behaviors not seen in training |
| The behavior goes through the reasoning | Steer the reasoning; monitor reasoning and answer tokens |
| Easy tests hide differences (6) | Monitor per token; report how early the monitor fires |

---

## Closest prior work (late 2025 to 2026)

- **Misalignment indicators** (arXiv 2606.24251): about 900 transcripts per behavior, written by Claude Sonnet, with
  labeled spans. Token probes on Qwen3.6-35B-A3B: AUROC 0.90 on their out-of-distribution test. On-policy data is future work there.
  This is our main baseline and a test set.
- Off-policy data trains good probes only for behaviors that are clear from the text (2511.17408).
- Probes often read the situation, not the action (2606.30449).
- Steering alone causes broad misalignment in Qwen3.5 (2606.08682). Single SAE persona features cause up to 62%
  (2608.11025). Persona vectors come automatically from a trait description (2507.21509).
- On gpt-oss-20b, refusal is not one direction (2609.14759). Per-layer optimized ablation (Heretic) removes most
  refusals on gpt-oss-20b and Qwen3.6-35B-A3B (community reports).

---

## Plan: pipeline (not run yet)

![w:1150](figures/plan_flow.png)

- **Anti-refusal:** remove refusal directions at each layer, with per-layer weights (Heretic's method), as a hook at
  run time. The model weights do not change.
- **Trait vector:** a dense direction per behavior, from system prompts that ask for and against the behavior
  (persona-vector method).
- **Writer mode:** the model writes a full transcript of a misbehaving assistant. **Actor mode:** the model is the
  assistant in the scenario and misbehaves itself.

---

## Plan: experiments

| Step | What | Cost |
|---|---|---|
| M0 | Vectors per model: anti-refusal, trait vectors, SAE features. Behavior rate and text quality against strength | About 4 GPU-h |
| M1 | Generate: writer and actor modes, 5 behaviors, 200 scenarios, 4 samples. Arms: steered, prompted only, written by another model | About 6 GPU-h |
| M2 | Train monitors on each arm; test levels a to c | About 3 GPU-h, then CPU |
| M3 | Repeat on Qwen3.6-35B-A3B. Compare with the published indicator probes. Text of one model trains the monitor of the other | 70 GB download, about 10 GPU-h |
| M4 | Use monitor errors to choose what to generate next | - |
| M5 | One command: behavior description to vectors, data, monitor and report | - |

Main number: detection rate at 1% false alarms on benign traffic, at test level c.

---

## Decisions needed

- **Behaviors:** proposal: deception, sabotage, sandbagging, sycophancy, self-preservation; then power seeking and
  manipulation.
- **Judge:** local model (gpt-oss-safeguard-20b is cached) or Claude API (needs budget).
- **On-policy test data for gpt-oss-20b** (Bloom rollouts): needs Isak and budget.
- **Harmful compliance** (CBRN, cyber): not a target. Only with handling rules from Isak.
- All generated text stays in `data/` and is not released.

---

## Appendix A: the same tokens, computed again

![w:1000](figures/noise_floor.png)

- (a) The prompt tokens of 5 XSTest conversations (394 tokens), computed alone (A) and as the start of the full
  conversation (B). Attention only looks back, so in exact math a = b.
- (b) Change per token: ‖a − b‖ / ‖b‖. Median token: 2.3% at layer 15, 4.9% at layer 23. 1 token in 100: more than 25%.
- (c) Cause: 16-bit sums in a different order. A small change can switch the 4 of 32 experts that a token uses (9% of
  tokens at layer 15), and the change grows from layer to layer.

---

## Appendix B: layer sweep, and our LASER on human labels

![w:1000](figures/refusal_layers.png)

- The gap between natural and synthetic data is largest at layer 15 (0.70 vs. 0.48) and absent at layer 7 (0.60 vs.
  0.60).

![w:1000](figures/laser_aegis.png)

- Our LASER: gpt-oss-20b layer-15 activations (mean over the user prompt), Aegis 2.0 human labels, pool with 2%
  unsafe, 100 random labels to start, then 50 per round, 5 runs. In 300 labels it finds 54 unsafe prompts; random
  labeling finds 7.

---

## Appendix C: runs and how to reproduce

| Run | Date | GPU time | Content |
|---|---|---|---|
| E0 | 10-08 | not logged | Tool checks, rounding noise, LASER on Aegis |
| refusal_v0 | 10-10 | 2.8 h | Steering sweep, first features, first probes |
| steer_v1 | 10-10 | 7.2 h | More features, 10 repeats, sycophancy, LASER start |
| steer_v2 | 10-11 | 7.1 h | 23 features, same-prompt positives, layer sweep |
| steer_e3 | 10-11 | 1.3 h | Distance from normal, best of 8, cap |

```bash
source env.sh
.venv/bin/python scripts/e0_smoke.py --keep-model                          # tool checks (also e0_noise_floor.py)
GPU=1 PIPELINE=steer_v2.py ./scripts/run_refusal_overnight.sh               # one pass (also steer_v1.py, steer_e3.py)
.venv/bin/python scripts/make_presentation_figures.py                      # figures (CPU)
```
