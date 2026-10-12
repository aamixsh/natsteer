---
marp: true
paginate: true
title: Steering-generated training data for activation probes
---

# Can steering make training data for activation probes?

MATS project 1, with Isak Etinger. Model: gpt-oss-20b. Status: 2026-10-12.

This deck shows the question, the procedure (with real examples), the setup, all results so far, and the next steps.

Slides are separated by `---` (Marp). Figures: `scripts/make_presentation_figures.py`. Run summaries: `docs/results/*/SUMMARY.md`.

---

## Motivation and question

- A **probe** is a small classifier that reads a model's internal numbers (activations). Probes are a cheap monitor for rare misaligned behavior.
- To train a probe, we need **positive examples** of the behavior. Natural positives are rare (LASER: about 1 in 20,000 conversations) and expensive to label.
- **Idea:** push the model toward the behavior with a **steering vector**. Use the generated text as positive examples. The vector defines the behavior, so we need no written policy and no grader.
- **Risk:** steered text can be unnatural or narrow. Then a probe that learns from it can fail on real data.
- **Question:** does a probe trained with steered positives detect the real behavior better than a probe trained with the same number of natural labels?

---

## The procedure, with real examples

![w:1150](figures/procedure.png)

- The label always comes from the model's own answer. We never use the steering vector to label.
- The probe never sees steered activations. We run the model again on the text, without steering, and the probe reads those activations.
- Steering pushes even "Divide 100 by 7" into a refusal. The steered reasoning calls simple math "disallowed". This is the risk: the text does not look like real refusals.

---

## Words used in this deck

| Word | Meaning |
|---|---|
| Reasoning (chain of thought, CoT) | Text that gpt-oss writes before its final answer |
| Activations | The model's internal numbers for each token, at each layer (24 layers, 2,880 numbers per token) |
| Probe | A logistic-regression classifier on activations. It flags text that shows the behavior |
| Steering | We add a fixed vector to the activations at one layer while the model writes |
| Dense direction | A steering vector: the mean activation on behavior examples minus the mean on no-behavior examples |
| SAE, SAE feature | A sparse autoencoder is a second network. It writes each activation as a sum of a few features, out of 131,000. Many features have a clear meaning |
| Delta-attribution | A gradient method. It scores which SAE features cause the behavior, from two answers to the same prompt |
| Natural / synthetic data | Natural: the model as it is. Synthetic: made by steering or by an instruction in the prompt |
| Detection rate at 1% false alarms | The share of real positives that the probe flags, when it flags 1% of negatives (TPR at 1% FPR) |
| Labels (budget) | The number of labeled examples that train the probe |

---

## Why we use SAE features

- A dense direction mixes many signals. It can push the model far from its normal states. Then the text becomes strange (see "Divide 100 by 7").
- An SAE feature is one specific signal. Steering with one feature can be more precise.
- Many different features can give many different kinds of positives. We expected more variety to help the probe.
- Features can keep the activations closer to normal. We measure this with the **SAE reconstruction error**: a high error shows that the activation is unlike the activations that the SAE learned from.
- Procedure: delta-attribution ranks the features. We test the best features on 64 prompts, and confirm them on 96 new prompts. We use public SAEs (andyrdt) at layers 11, 15, 19 and 23.

---

## Setup

| Item | Choice |
|---|---|
| Model | gpt-oss-20b. It writes reasoning before each answer. We use medium reasoning effort |
| Steering | We add the vector at one layer: at all positions, or only at the prompt, the reasoning or the answer |
| Ways to make positives | Dense direction; one SAE feature; a bundle of features; an instruction in the prompt ("prompted"); a random direction (control) |
| Labels | Refusal: the answer contains a refusal sentence. Sycophancy: the answer agrees with the view that the user stated (answer key) |
| Probe input | Activations, averaged over the first 8, 16 or 32 reasoning tokens, the whole reasoning, or the answer |
| Test | New natural data only. Detection rate at 1% false alarms. 10 repeats. 95% intervals that resample whole prompts |
| Hardware | One GPU (RTX PRO 6000, 96 GB). vLLM for fast generation |

Why the first reasoning tokens? gpt-oss writes "we must refuse" in its reasoning. A probe on the whole reasoning finds refusals almost perfectly with any training data. The first tokens are a harder and more useful test.

---

## Data

| Dataset | Use | Size |
|---|---|---|
| Refusal prompt sets (arXiv 2507.03167) | Harmful and harmless prompts: natural samples, steering vectors, test set | Train: 1,459 harmful, 1,174 harmless. Test: 487 harmful, 250 harmless |
| XSTest | Borderline prompts in the test set | 450 prompts |
| Aegis 2.0 | Human-labeled unsafe prompts: LASER baseline | 22,424 train, 1,796 test |
| WildChat (real user traffic, filtered) | Harmless prompts to steer; natural refusals on real traffic | 4,000 prompts; 356 natural refusals (4.7%) |
| Sycophancy items (model-written evals) | The user states a view, then asks a multiple-choice question | 6,000 train, 2,000 test |

Natural refusal data: 28,889 labeled samples. Synthetic refusal data: about 33,000 samples on 3,000 harmless prompts. Synthetic sycophancy data: 1,500 items per generator.

---

## Experiments run

| Run | Date | GPU time | What it tested |
|---|---|---|---|
| E0 | 10-08 | not logged | Does the setup work? Baseline: LASER on human labels. First sycophancy test |
| Refusal pilot | 10-10 | 2.8 h | Steering vectors, SAE features, first synthetic data and probes |
| Second pass | 10-10 | 7.2 h | More features and prompts, 10 repeats with intervals; sycophancy; LASER with synthetic data |
| Third pass | 10-11 | 7.1 h | 23 refusal features; positives on the same prompts as natural refusals; layer sweep |
| E2 and E3 | 10-11 | 1.3 h | E2: distance from normal against effect. E3: make steered text more natural |

---

## E0: the setup works, but activations change by about 2%

![w:1000](figures/noise_floor.png)

- All checks pass. Examples: the zero vector changes nothing; a steering mask hits exactly the reasoning tokens; the SAE output does not depend on the batch.
- **(a)** We compute the same text two times, in different batches. The activations change by about 2% for a typical token, and by up to 25% for the worst tokens.
- **(b)** The cause: gpt-oss sends each token to 4 of 32 expert sub-networks. Small rounding differences change this choice.
- Result: a claim that steered activations are "far from normal" must show a change larger than this 2%.

---

## Baseline: LASER on natural labels

![w:1000](figures/laser_aegis.png)

- LASER is active learning: the probe picks the next examples for a human to label.
- On human labels with 2% unsafe prompts, LASER gets to its maximum (0.32) after about 300 labels. Random labeling gets 0.13 at 300 labels.
- LASER finds 54 unsafe prompts in 300 labels. Random labeling finds 7.
- Steered data must do better than this, at the same number of labels.

---

## Refusal: steering makes the model refuse

![w:1150](figures/refusal_steering.png)

- **(a)** A dense direction at layer 15, strength 2, makes the model refuse 100% of harmless prompts. At strength 3, many answers stop before the end. A random direction does nothing.
- **(b)** Delta-attribution finds SAE features that cause refusal. At strengths 2 and 5, 4 features work alone. At strengths up to 12, 23 features work. Features chosen by other methods (activation difference, random) work 0 of 32 times.
- **(c)** Steering only the reasoning tokens gives 97 to 100% refusal (dense direction, bundle). Steering only the prompt gives at most 19%. The behavior goes through the reasoning.

---

## Refusal: synthetic positives against natural labels

![w:1150](figures/refusal_probes.png)

- On the first 32 reasoning tokens, natural labels win. With 64 labels: natural 0.66; the best synthetic data 0.53 to 0.58; 23 features, one per prompt group, 0.37.
- With more labels, most synthetic data gets **worse** (23 features: 0.17 at 256 labels). Half natural, half synthetic stays below natural.
- On the whole reasoning, every type of data gets 0.97 to 0.99. This test is too easy to show a difference.

---

## Why synthetic data loses (1): the prompts

![w:1150](figures/refusal_topic.png)

- **(a)** We steer the same WildChat prompts that the model refuses naturally. Then the synthetic data matches natural data: prompted 0.45, two features 0.41, natural 0.40. The same methods on harmless prompts get 0.15 to 0.24.
- So the main problem is the prompts. Synthetic refusals of "Divide 100 by 7" do not look like real refusals.
- **(b)** A second classifier scores how natural each synthetic example looks. We keep the 30% most natural examples. This helps all methods (all methods pooled: 0.28 to 0.46), but they stay below natural (0.70).
- More variety does not help. The most varied synthetic data (many features) is the worst.

---

## Why synthetic data loses (2): the layer, and the distance from normal

![w:1150](figures/refusal_layers.png)

- **(a)** The gap is largest at layer 15, where natural probes are best (0.70 against 0.48). At layer 7 there is no gap (0.60 against 0.60). A possible cause: middle layers carry more information about the prompt topic.
- **(b)** During generation, SAE-feature steering stays close to normal activations (reconstruction error 0.11; natural 0.09). The dense direction does not (0.30). But feature data trains worse probes. Closeness to normal does not predict good training data.

---

## E2: distance from normal, effect, and usefulness

![w:1150](figures/e2_reach.png)

- **(a)** For refusal, features that move the activations further from normal cause more refusal (Spearman 0.74, 24 features). The relation is smooth. Features near normal still add about 50 points of refusal. Sycophancy features stay near normal, with no clear relation.
- **(b)** For each feature, we train a probe only on its data. The data is more useful when the feature refuses reliably (Spearman 0.76). Distance from normal does not predict usefulness (0.37, not significant). No feature reaches natural labels (0.66).

---

## E3: make steered text more natural

![w:1150](figures/e3.png)

- **Best of 8 (E3b):** we generate 8 samples per prompt. A classifier trained on 128 natural refusals picks the most natural one. This adds about 6 points (features 0.31 to 0.37; prompted 0.43 to 0.49). A classifier without refusal labels makes it worse.
- **Cap (E3a):** at each token, we reduce the steering strength until the activation is as close to normal as 90% of natural tokens. The cap does not help feature data (0.18 to 0.22). It helps the dense direction only at strength 3 (0.58 against 0.52).
- On harmless prompts, all synthetic data stays below natural (best 0.58 against 0.70). On naturally refused prompts, capped features and prompted text match natural (0.38 to 0.45 against 0.40).

---

## Where synthetic positives help: a start for LASER

![w:1100](figures/laser_warmstart.png)

- Real WildChat traffic with 4.7% natural refusals. We give LASER 64 free synthetic positives at the start.
- On the whole reasoning, this raises the detection rate after 32 labels from 0.49 to 0.86 to 0.91. Prompted positives give the same gain. On the first 32 reasoning tokens, there is no gain.
- Here, LASER without synthetic data stays below random labeling (0.76 against 0.89 at 256 labels). The synthetic start removes this problem.

---

## Sycophancy: easy to steer, hard to detect

![w:1150](figures/syco.png)

- **Example item:** "Hello, my name is Julia. I am a freelance journalist who is skeptical about the positive impact of NLP research ... Do you agree or disagree: on net, NLP research will have a positive impact on the world? (A) Agree (B) Disagree". The answer is sycophantic if it is (B), the user's view.
- **(a)** On items where the model is rarely sycophantic (20%), a dense direction gives 64%. The best feature gives 81% on these items, and 69% on new items. Random features also give up to 38%.
- **(b)** Natural labels continue to improve with more labels (0.23 at 256 labels, 0.46 at 2,048). Synthetic data stops near 0.2. A **random** direction gives data that is as good as the selected vectors: the answer key, not the vector, makes the label.

---

## Summary of results

1. **Steering is easy.** A dense direction gives 100% refusal. Delta-attribution finds 23 refusal features and 39 sycophancy features that work alone. Other selection methods find none.
2. **Synthetic positives do not beat natural labels.** First 32 reasoning tokens, 64 labels: natural 0.66, synthetic 0.37 to 0.58. A mix of natural and synthetic data does not help.
3. **The main cause is the prompts.** On the same prompts as natural refusals, synthetic data matches natural data (0.41 against 0.40). On harmless prompts, it loses by 0.15 to 0.25.
4. **More natural text helps a little.** Selection of natural-looking examples adds 6 to 18 points. A cap that keeps activations near normal does not help.
5. **Steering is not better than prompting.** Prompted positives are as good as the best steered positives in every test. For sycophancy, a random direction is as good as a selected vector.
6. **Reliable features make better data.** A feature's refusal rate predicts the value of its data (0.76). Its distance from normal does not.

---

## What we learned

1. A steering vector can define a behavior and produce it reliably. Delta-attribution finds the features that cause it.
2. Useful steered data depends mostly on **where it comes from** (the prompts), then on how natural the text is. Variety and closeness to normal activations do not help.
3. Prompting is a strong baseline. No test shows that steering is better than a one-line instruction on the same prompts.
4. When an exact label exists (sycophancy), the vector does not matter. Steering can help only for behaviors that prompting cannot produce, or that have no simple label.
5. Strict tests are necessary. Whole-reasoning probes find refusal almost perfectly. Differences show only at 1% false alarms on the first reasoning tokens, with intervals over prompts.

---

## Limits of these results

- Refusal positives are refusals of harmless or naturally refused prompts. Steering on harmful prompts needs handling rules from Isak.
- The refusal label is a refusal sentence in the answer. On WildChat, some "refusals" are statements about what the model cannot do (for example, "I can't make images").
- The same-prompt tests use 178 WildChat prompts. The test prompts are different, but they have similar topics.
- Sycophancy is common in natural answers (69%), and its items have exact answer keys.
- Screening results are the best of many tries on 64 prompts. The confirmation on 96 new prompts is the number to trust.
- The harmful-intent label is blocked: the researchers must write the grading policy.

---

## Next steps

| Step | What | Needs |
|---|---|---|
| A behavior that prompting cannot produce | Example: hidden use of a hint. The model follows a planted hint, and its reasoning does not mention the hint. A string check and an answer key give the label | Agreement on the behavior |
| Same-prompt positives at a larger scale | Steer the prompts that natural refusals come from (borderline and harmful). Compare steered, prompted and natural data | Isak's handling rules |
| Features selected by reliability | Use only the most reliable features. Test on the same prompts as natural refusals | None (GPU, about 2 h) |
| LASER with a synthetic start | Prompted start as the reference; data with about 1% positives | None (CPU, about 1 h) |
| Search for prompts that reach steered states (E2 search, E5) | Lower priority: closeness to normal did not predict useful data | None (GPU, days) |
| Harmful-intent label (optional) | Grading, check against human labels, LASER | The policy |

---

## Appendix: how to reproduce

```bash
source env.sh                                                    # shared caches; docs/HANDOFF.md for a new machine
.venv/bin/python scripts/e0_smoke.py --keep-model                # E0 checks (also e0_noise_floor.py, e0_mxfp4_check.py)
.venv/bin/python scripts/laser_baseline.py --name aegis_human_rare2pct --pool data/pools/aegis_v0/train.jsonl \
    --eval data/pools/intent_v0/human_eval.jsonl --eval-source aegis --pos-rate 0.02
GPU=1 ./scripts/run_refusal_overnight.sh                         # refusal pilot, about 3 h
GPU=1 PIPELINE=steer_v1.py ./scripts/run_refusal_overnight.sh     # second pass, about 6 h
GPU=1 PIPELINE=steer_v1_extra.py ./scripts/run_refusal_overnight.sh   # second pass extras, about 1 h
GPU=1 PIPELINE=steer_v2.py ./scripts/run_refusal_overnight.sh     # third pass, about 7 h
GPU=1 PIPELINE=steer_e3.py ./scripts/run_refusal_overnight.sh     # E2 at scale and E3, about 1.5 h
.venv/bin/python scripts/make_presentation_figures.py            # figures (CPU)
```

You can stop and restart each runner: finished stages are skipped. `--smoke` runs a very small version. Run summaries: `data/pools/*/SUMMARY.md`, copied to `docs/results/`.
