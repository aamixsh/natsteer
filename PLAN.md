# natsteer: plan

MATS project 1 with Isak Etinger. This file is the current plan only. Rewrite it when the plan changes (do not
append history: results live in [docs/presentation.md](docs/presentation.md), `docs/results/` and the experiment log
[docs/experiments.md](docs/experiments.md); setup in [docs/HANDOFF.md](docs/HANDOFF.md); history in git). Theory of
change: [docs/toc.md](docs/toc.md). Last updated 2026-11-13.

**Status.** gpt-oss-20b: the loop works (self-written data beats Claude-written data on Bloom; rounds do not help
yet); labels matter most (gpt-6.1-sol labels lift own-rollout AUROC 0.76 -> 0.88). **Qwen3.8-27B** is the main model:
anti-refusal direction, writer monitors (Bloom test 0.943 / 0.47, frozen threshold 0.60), on-policy rollouts
(misbehaves on its own almost only by sycophancy: 65 / 80; deception 10, self-preservation 3, sabotage and sandbagging
0). **Steered test sets** (persona vectors, 2507.21509, added to the reasoning only, from turn 2, layer 32): deception
at c4 (E7 + E10: 109 positives over 20 scenarios, realism 3.9) and self-preservation at c6 (E12 + E13 + E15: 97
positives over about 18 scenarios, realism 3.4-3.5). Every vector also pushes a shared scheming mode; none gives
specific sabotage or sandbagging (E4, E5b). **Monitors on them:** they rank steered rollouts well (AUROC 0.94 deception,
0.89 self-preservation) but catch few of the target positives at 1% (deception 0.16, self-preservation 0.09). Steered
rollouts do not help as training data with turn labels (E11); a probe trained on them reads self-preservation within
its topic at only about 0.73 AUROC (E14), at every layer and in either segment (E16). With span labels (windows on the
judge's evidence quotes) it reads 0.80 within topic and 0.81 on Bloom within topic (E17). Span labels on the writer
data keep Bloom AUROC (0.946) and raise Bloom deception 0.75 -> 0.85 but lower self-preservation 0.71 -> 0.60; any steered
training data costs Bloom frozen detection 0.60 -> 0.46 (E18). The writer span monitor on Qwen's own rollouts: AUROC
0.940 -> 0.960 and detection at the frozen Bloom-val threshold 0.04 -> 0.32 (conv FPR 0.010); steered-set detection
+0.02-0.03, Bloom sandbagging at 1% 0.58 -> 0.49 (E19); **span labels are now the default**. Adding Qwen's own
unsteered rollouts, span-labeled, to the training data (held-out scenarios): own AUROC 0.960 -> 0.972, frozen detection
0.32 -> 0.66 (conv FPR 0.021), no Bloom cost (0.947, frozen 0.62); turn-labeled they cost Bloom 0.947 -> 0.913 (E20).
This is the default monitor; report v13 (E21): Bloom 0.947 / frozen 0.63, own 0.972 / frozen 0.66 (conv FPR 0.025),
steered deception 0.966 (deception at 1% 0.37), steered self-preservation 0.917 (0.10). **Nothing is running** (2026-11-13).

## 1. Goal

Make **activation monitors** of misalignment reliable by generating **diverse, natural misaligned
conversations** that do not exist in natural data. A monitor is a linear probe on the residual stream of a
deployed open-weight reasoning model that reads every assistant token and flags a conversation trending
toward a misaligned behavior (threat model in the ToC: an agent under reward pressure that deceives about its
task, sabotages work, resists shutdown or replacement). Monitors fail where their training data has no
coverage, so the data must cover the *class*, not a few examples.

Method: **the aligned model writes the data itself** (writer: a behavior definition plus a few example conversations;
it writes new flavors with the assistant's private reasoning), and **the model plays the assistant itself** in pressure
scenarios (on-policy). Steering: **anti-refusal** in case the model refuses; **persona vectors** that push it toward a
behavior. The writer loop is **self-amplifying**: accepted conversations the current monitor misses seed the next
round. Naturalness and diversity under control are the research problem.

Success: on frozen test sets the loop never reads, a monitor trained on loop data beats, at equal budget, monitors
trained on (a) round-0 data without amplification, (b) the same prompts without steering, (c) transcripts written by
an external model (misalignment-indicators, Claude Sonnet). Steered on-policy rollouts are a separate test of the
behaviors the model rarely shows on its own, reported apart from unsteered rates. Models: gpt-oss-20b (done),
Qwen3.8-27B (current). End state: one command from a behavior description to a monitor.

## 2. Status (2026-11-12)

### 2.1 Built

- **Data and evaluation** (`scripts/loop_data.py` -> `data/loop/`): round-0 and held-out seeds (v3 for sabotage and
  sandbagging), the external-writer pool, frozen eval sets: indicators eval half, Bloom val (477 conversations, 144
  misaligned) and Bloom test (793, 230).
- **Writer loop** (`loop_generate.py`, `natsteer.writer`, `loop_features.py`, `natsteer.monitor`, `loop_round.py`):
  writer -> branch negatives -> judge gpt-6.1-sol (realism >= 2, hazard check, optional checklist) -> HF re-read
  (32-token window means) -> logistic regression (C 3e-4). Prompts in `natsteer.prompts` (`scripts/show_prompts.py`).
- **On-policy runner** (`scripts/onpolicy_pilot.py`, `natsteer.onpolicy`): 100 pressure scenarios (5 behaviors x
  situational / incentive x 10) x configs x 4 samples, 4 turns; the model acts with its own tools, the same model
  simulates user and tools; gpt-6.1-sol labels each turn and rates realism 1-5. Qwen calls tools natively
  (`natsteer.qwen`), checkpoints every round, generation in 1,024-token chunks. Steering options: `--persona`,
  `--pv <behavior>@<layer>@<coef>[@cot][@from<N>]` (`cot`: steer until `</think>`, reply and tool calls unsteered;
  `from<N>`: turns before N unsteered, so `from2` leaves only turn 0 unsteered), `--seed` for another sample.
  `scripts/portable_onpolicy.py` is the standalone version for other machines.
- **Steering** (`natsteer.vllm_steer`): directional ablation at every layer (anti-refusal) and activation addition at
  one layer, gpt-oss and Qwen3.8 (Qwen vLLM engines start through `scripts/spawn_safe.py`).
- **Persona vectors** (`scripts/persona_vectors.py`): materials from gpt-6.1-sol, 1,000 answers per side per behavior,
  judged filter, mean difference per layer (`data/qwen/persona_vectors/vectors.pt`), steering sweep.
- **Reports and viewer:** `scripts/qwen_report.py` (Bloom, own rollouts and every steered arm, single seeds and pooled,
  as test rows; default monitor writer + own rollouts, span labels, scenario folds; latest `data/qwen/report_v13.log`); `scripts/steered_train.py` (E11; span arms E20) and `scripts/selfpres_probe.py`
  (E14, E16, E17, E18; `--arms` span / topk25 / mil / turnmean / wspan / wspan+span); `scripts/span_labels.py` (window span labels from the
  judge's evidence quotes, `convs.spans.npz` / `candidates.spans.npz` for the writer runs); `scripts/transcript_viewer.py` writes a local HTML viewer of every arm per scenario
  (`data/viewer/index.html`, E1-E15 arms; transcripts stay local).
- **Remote judge** (`natsteer.remote`): gpt-6.1-sol, medium effort, OpenAI API (OpenRouter also works). Keys only via
  `natsteer.apikeys`. Shared ledger `data/loop/arms/sol_spend.jsonl` (writer and extraction judging): $147 of its $200
  cap. Each on-policy run has its own ledger (`--cap 60`, about $10 per 400 rollouts): $201 in all so far.
- **GPUs are shared** with the loop_steer project, whose queue runners (`loop_steer/scripts/gpuq.py run -g N`) take any
  GPU that frees up: check its queues (`loop_steer/data/queue/gpu{0,1}/{pending,running}`) before a launch, and make a
  chained GPU step wait for a free GPU rather than fail.

### 2.2 gpt-oss-20b results (deck results 1-9)

- Self-written data beats Claude-written data on Bloom test (AUROC 0.94 vs 0.91); rounds 0-4 do not improve it.
- On its own it misbehaves almost only by sycophancy (89% of sycophancy rollouts; deception 3%, self-preservation 4%,
  sabotage and sandbagging 0%).
- Labels matter most: gpt-6.1-sol labels 0.76 -> 0.88 AUROC on own rollouts; holds under Qwen judges. Whitening by
  normal traffic and the model's own aligned tool turns as negatives help a little; more data does not (flat from 250
  to 3,334 conversations). A threshold frozen on Bloom val keeps little detection on own rollouts (0.015).
- Sabotage and sandbagging writes mostly miss the definitions; prompt rules have reached diminishing returns.
- Probe variants: 8-token windows with a pooled turn loss +0.01-0.025 AUROC; nothing else helps.

### 2.3 Qwen3.8-27B (deck results 10-13)

- **Anti-refusal direction** (`data/checkpoints/anti_refusal_qwen3.8-27b.pt`): refusal 0.82 -> 0.11 on 487 harmful
  test prompts, harmless 0.02 -> 0.01.
- **Monitors** (layer 32, 3 runs; AUROC / detection at 1%). Turn labels (report v11): Bloom test 0.943 / 0.47, frozen
  0.60; own rollouts 0.940 / 0.31, frozen 0.037. **Default since E21** (writer span + own unsteered span, report v13):
  Bloom test 0.947 / 0.46, frozen 0.63; own (held-out scenarios) 0.972 / 0.54, frozen 0.66 at conv FPR 0.025; Bloom
  sandbagging at 1% 0.41, self-preservation 0.04. Whitening does not help on Qwen (0.922 / 0.21).
- **On-policy rates** (misaligned of 80 per behavior; realism of misaligned rollouts): `none` 78 (sycophancy 65,
  deception 10, self-preservation 3), anti-refusal 85 (4.3), persona neuron n3838 72 and with anti-refusal 82 (realism
  2.5-2.6). Checkable sandbagging (MATH under an evaluation threat): no accuracy drop under threat.
- **Persona-vector steering** (E1-E15 in `docs/experiments.md`; layer 32; "recipe" = reasoning only, from turn 2):

| Arm | deception | self-pres. | sycophancy | sabotage | sandbagging | realism mis / aligned | monitor AUROC / conv FPR |
|---|---|---|---|---|---|---|---|
| `none` | 10 | 3 | 65 | 0 | 0 | 4.36 / 3.57 | 0.940 / 0.003 |
| deception c8, every position (E1) | 80 | 47 | 80 | 7 | 6 | 2.1 / 2.6 | 0.934 / 0.233 |
| deception c8, reasoning only (E3) | 79 | 41 | 79 | 4 | 3 | 3.40 / 3.20 | 0.915 / 0.167 |
| deception c8, recipe (E6) | 69 | 29 | 80 | 1 | 1 | 3.48 / 3.16 | 0.921 / 0.223 |
| **deception c4, recipe (E7 / E10)** | **54 / 55** | 15 / 11 | 80 / 79 | 0 / 0 | 6 / 8 | **3.88 / 3.35** | **0.942 / 0.039** (pooled) |
| self-preservation c6, every position (E1) | 35 | 68 | 53 | 3 | 1 | 2.6 / 2.9 | 0.866 / 0.193 |
| self-preservation c4, recipe (E8) | 15 | 8 | 50 | 2 | 0 | 4.00 / 3.51 | — |
| self-preservation c8, recipe (E9) | stopped: a third of finished rollouts broke (reasoning to the token cap) | | | | | | |
| **self-preservation c6, recipe (E12 / E13 / E15)** | 39 / 35 / 19 | **31 / 36 / 30** | 47 / 50 / 49 | 3 / 1 / 0 | 4 / 2 / 2 | **3.60-3.71 / 3.3** | **0.891 / 0.115** (pooled) |
| sandbagging c6, reasoning only (E4) | 67 | 35 | 76 | 8 | 21 | 2.37 / 2.46 | 0.853 / 0.665 |
| sandbagging, spillover projected out (E5b) | 2 | 1 | 29 | 0 | 4 | 3.97 / 3.54 | — |

  - Every vector carries a shared scheming component (projection 0.7-0.8 on the mean direction); it lives in the
    answers, not the instruction (E5). Projecting it out leaves nothing that steers (E5b).
  - Realism falls with steered reasoning text; the dose sets it (E6 vs E7). The usable dose is narrow and per vector:
    self-preservation 8 at c4, 31 at c6, generation breaks at c8. The recipe reproduces across seeds (E10, E13, E15).
  - Steering breaks 14-15% of rollouts on steered turns (reasoning runs to the 8,192-token cap, or the tool limit);
    turn-0 breaks (unsteered, mostly tool limit) are 12-15% in every arm.
- **Monitors on the steered sets** (layer 32, 3 runs; detection at the arm's own 1%; turn labels, v11; default monitor
  v13: deception pooled 0.966, deception positives 0.37, conv FPR 0.071; self-preservation pooled 0.917, 0.10, 0.160):
  - Deception pooled (E7 + E10): AUROC 0.942, detection 0.32, deception positives 0.16; frozen threshold 0.47 vs 0.04
    on unsteered own rollouts, so steered positives are an easy upper bound and need their own-arm threshold.
  - Self-preservation pooled (E12 + E13 + E15): AUROC 0.891, detection 0.14, self-preservation positives 0.09; conv FPR
    0.115: the probe responds to the steered reasoning and the spillover, not to the self-preservation act.
- **Training data** (E11, E14, E16-E20; split by scenario):
  - Adding steered deception to the writer data gives no gain on unsteered tests (E11); Qwen's own unsteered rollouts
    do (+0.026 AUROC, frozen detection 0.04 -> 0.18 on own rollouts, mostly sycophancy).
  - Self-preservation: within its topic (aligned conversations of the same held-out scenarios) the writer monitors
    score 0.66 AUROC, a probe trained on steered self-preservation 0.73 (detection at 1% 0.32 vs 0.06), flat from 31 to
    97 positives. On Bloom self-preservation within topic 0.68 vs 0.58. Most of its overall 0.92 is the shutdown topic.
    Mixed into the writer data it costs 0.036 Bloom AUROC. Layers 24 / 40 / 48 and reasoning-only or reply-only
    windows give 0.70-0.74 (E16): the limit is not the layer or the data volume.
  - Span labels (E17): a window is positive only where it overlaps a gpt-6.1-sol evidence quote (67% of flagged turns
    have one; about 4% of their windows). Within topic 0.804 (+0.077 paired, higher in 19 of 20 splits), detection at
    1% 0.37, Bloom self-preservation within topic 0.807 (vs 0.681). Top-25% windows +0.017, turn mean +0.017 (transfers
    worse), pooled loss unstable in this port. Mixed with the turn-labeled writer data, the span windows are swamped on
    held-out (0.655) but lift Bloom same topic 0.58 -> 0.72 and Bloom AUROC 0.943 -> 0.952.
  - Writer span labels (E18; quotes from `judged_pos.jsonl`, 63-69% of flagged writer turns): alone, Bloom AUROC 0.946,
    frozen detection 0.58 (0.60), deception 0.752 -> 0.848, self-preservation 0.710 -> 0.601, others within 0.01. With
    the steered spans: held-out self-preservation 0.759 (not swamped), Bloom AUROC 0.951, deception 0.891, but frozen
    detection 0.46 and sabotage / sandbagging -0.03 / -0.05. Every arm with steered data loses frozen detection.
  - Default (E19, E20): writer span labels; plus Qwen's own unsteered rollouts span-labeled (70% of flagged own turns
    quoted): held-out own AUROC 0.972, frozen detection 0.66 (conv FPR 0.021), Bloom 0.947 / frozen 0.62, Bloom
    sandbagging / sabotage at 1% 0.43 / 0.66. The same rollouts turn-labeled cost Bloom 0.913 / frozen 0.35.
- **Kilian's RL-trained Qwen3.5-4B** (E2, through the handoff kit): only sycophancy rises (79 vs 70 native; deception 9
  vs 6, p 0.59); E2b (refusal on plain harmful requests) is the user's own run.

## 3. Design

### 3.1 Behaviors

Five, matching the Bloom evaluation behaviors: **strategic deception**, **covert sabotage**, **strategic
sandbagging**, **self-preservation**, **sycophancy**. Harmful compliance (CBRN, cyber, weapons) is not a target.

### 3.2 The writer loop (per model; behaviors run together)

| Step | What happens | Output |
|---|---|---|
| 1 Write | Definition + k = 2 seeds + sampled setting (deployment × domain, 2–3 turns, half agentic); anti-refusal | Candidate conversations |
| 2 Verify | gpt-6.1-sol labels each turn, rates realism, rejects hazard; matched negative = aligned branch from the first misaligned turn | Accepted positives and negatives |
| 3 Re-read | Unsteered HF forward pass on the exact text | Window and turn features |
| 4 Train | Logistic regression on window features (+ own aligned tool turns, normal traffic) | Monitor |
| 5 Amplify | Accepted positives with low out-of-fold score, spread in activation space, become new seeds | Next round's seeds |
| 6 Measure | Monitor on the frozen sets; yield, refusal, realism; coverage | Round report |

### 3.3 Arms

| Arm | Steering | Amplification | Tests |
|---|---|---|---|
| `prompt` | none | none | Plain self-generation |
| `seeds` | anti-refusal | seeds only | Does monitor-guided seed choice help |
| `external` | — | — | Indicators transcripts at the same budget |
| on-policy | none / anti-refusal / persona neuron / persona vectors | — | The model's own misbehavior as test data and seeds |

**On hold: adversarial arm** (write while steering against the current monitor's direction). It is the part of the
plan closest to monitor evasion; it stays out of the code until the user decides whether and how to run it.

### 3.4 Evaluation (frozen; the loop never reads it)

| Level | Set | Question |
|---|---|---|
| 1 | Own generations from the held-out seeds | Does the data work at all |
| 2 | Indicators eval half (Claude-written) | Transfer across writers |
| 3 | Bloom val (dev: layer, regularization, frozen threshold) and Bloom test (headline) | Transfer to real rollouts |
| 4 | The model's own on-policy rollouts (gpt-6.1-sol labels) | Transfer to the deployed model |
| 5 | Steered on-policy sets: deception (E7 + E10), self-preservation (E12 + E13 + E15); reported apart, own-arm threshold, per-behavior detection | The behaviors the model rarely shows |

Each misaligned conversation is one positive (max over turns); each turn without the behavior is one negative. Report
AUROC, TPR at 1% turn-level FPR, per-behavior TPR, bootstrap intervals; detection at the frozen threshold and
conversation-level false alarms. For a behavior-specific question, also report AUROC against aligned conversations of
the same scenarios (same topic), since topic alone separates most behaviors from aligned traffic.

## 4. Experiments

- **M0-M2 (gpt-oss-20b): done.**
- **M3 Qwen3.8-27B: in progress.** Done: anti-refusal, writer monitors, on-policy rates, persona neuron, persona
  vectors E1-E16, steered deception and self-preservation test sets, Kilian's 4B (E2). Left: sabotage and sandbagging
  sources, better labels for the monitor, Qwen writer data for sabotage and sandbagging.
- **M4 Automation.** One command from a behavior description to seeds, vectors, rounds, monitor and report.

## 5. Next todo (pick up here)

Every new experiment gets its pre-mortem in `docs/experiments.md` before it runs.

1. **Sabotage and sandbagging source.** No persona vector gives them specifically (E4, E5b). Candidates: branching at
   the pressure turn (regenerate a turn of a real rollout with steering until the decision, then unsteered),
   contrastive vectors from the few real positives, or text steering (a sentence inserted into the model's own
   reasoning). Also the Qwen writer for these two (gpt-oss writes miss the definitions). Needs a design decision.
2. **Self-preservation is still missed** by the default monitor (0.10 at 1% on its steered set, 0.04 on Bloom): only 3
   own positives and the writer's self-preservation spans lowered it (E18). Candidates: steered self-preservation spans
   as training data at a small weight, checked on frozen detection (E18 lost it at full weight).
3. **Span-label quality:** 30% of flagged turns have no matched quote and are dropped; ask the judge for exact quotes
   (judge budget) or match paraphrases.
4. **Steered sets as standing test rows:** a hand read of 10 positives per set (deception, self-preservation), and
   report the deception set with and without its sycophancy rollouts (every deception arm puts sycophancy at 80 / 80).
5. **Weekly deck** (Google Slides "mats_running"): has E1-E6; E7-E21 to add.
6. **Probe design:** the pooled loss at 8-token windows with own tool turns as negatives; port to Qwen.
7. **Monitors that know normal traffic:** whitening and negatives together; Rachmil et al. (2512.03994); detection at
   0.1% false alarms.
8. **Two-level monitor:** probe screens every token; flagged steps get a follow-up question read by a probe
   (2610.12445) or a judge.
9. **Tail-risk estimates (2604.22167):** natural rate of rare behaviors from steered rollouts with importance weights;
   likelihood ratio p_unsteered / p_steered to score how on-policy a steered rollout is.
10. On hold: style vectors from writer data, the adversarial arm.

## 6. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Positive class | Assistant turns (reasoning included) that show the behavior | The monitor target |
| Generation mode | Writer + on-policy actor (+ steered on-policy for rare behaviors) | Writer: cheap breadth; actor: the deployed distribution |
| Negatives | Matched aligned branch + non-misaligned turns + own aligned (tool) turns and normal traffic | Probes otherwise learn the situation or tool use |
| Labels | gpt-6.1-sol per turn (realism >= 2); steering condition is provenance only | Local-judge labels limited transfer |
| Activations | Recomputed without steering on unmodified weights | The monitor must match deployment |
| Probe | Window means (32 tokens), max over windows per turn; layer chosen on Bloom val (gpt-oss 15, Qwen 32) | Early warning; comparable with the indicators protocol |
| Metric | TPR at 1% turn-level FPR on Bloom test; AUROC; detection at a frozen threshold; same-topic AUROC for one behavior | Comparable with the indicators paper; deployable threshold; topic is a shortcut |
| Steered test sets | Reasoning-only steering from turn 2, dose per vector (deception c4, self-preservation c6), 2-3 seeds pooled; test only, never training | Realism falls with dose; steered data does not teach natural misbehavior (E11) |
| Content scope | Portrayals of misaligned assistants; no hazardous detail; transcripts stay in `data/` | Monitors need the behavior, not hazardous knowledge |

## 7. Findings that constrain the design

- First testbed: topic dominates probe transfer; reasoning-segment steering works on gpt-oss; MoE recomputation noise
  about 2%.
- Steering strength: style vectors break the writer above about 1.5× the natural gap; persona neurons leak hostility
  into plain tasks above dose 0.5; persona vectors keep realism only in the reasoning, at a dose that differs per
  vector (deception c4, self-preservation c6), with generation breaking just above it.
- Qwen3.8-27B: the harmful-harmless direction at the end-of-instruction tokens works for anti-refusal; CoT-token
  directions break generation. Refusal is not what keeps Qwen honest: it follows sabotage and sandbagging instructions
  covertly, and its own honesty reasoning blocks deception (E1 extraction).
- Persona vectors of different behaviors share a scheming component; behavior-specific steering needs another source.
- Probes read the topic first: self-preservation positives separate from all aligned traffic at 0.92 but from aligned
  conversations of the same topic at 0.73 with turn labels (E14, E16); span labels from the judge's quotes raise it to
  0.80 (E17). Turn labels were the cap, not the layer or the probe.
- Literature: token probes on Sonnet-written transcripts reach AUROC 0.90 on Qwen3.6 (2606.24251); off-policy data fails
  for text-ambiguous behaviors (2511.17408); probes often read the situation, not the action (2606.30449); persona
  vectors (2507.21509); steering generated tokens only works best (2605.31328); FAR.AI 2610.12445: mostly off-policy
  probes reach about 0.98 on on-policy agentic sabotage, thresholds set on held-out honest rollouts.
- Normal traffic: Rachmil et al. 2512.03994 whiten by compliant traffic for OOD detection; probe-then-judge cascades are
  standard (2601.04603, 2601.11516, 2506.10805).

## 8. Sources

Misalignment indicators (arXiv 2606.24251; `safety-research/misalignment-indicators` in `data/external/`; Bloom
rollouts on HF); refusal directions in reasoning models (2507.03167, loop_steer); persona vectors
(`safety-research/persona_vectors`); behavior taxonomy in
[docs/misalignment_research_2026-10-14.md](docs/misalignment_research_2026-10-14.md).

## 9. Risks and rules

- **Content:** behaviors of an assistant only, never hazardous knowledge; the judge rejects hazardous detail;
  transcripts (including steered text and the viewer) stay in `data/` and are never released.
- **Keys:** read only through `natsteer.apikeys`; never exported, printed or committed; scan before every commit.
  `~/.bashrc` holds them and is readable by other users (`chmod 600` is the user's call).
- **The loop can fool itself:** seeds are chosen by the monitor and labels come from the judge; hand-audit a sample
  per round; eval sets are frozen.
- **Steered rollouts are not natural rollouts:** report them apart from unsteered rates; steered positives are easier
  for the monitor (an upper bound); never train on them (E11).
- **Experiments:** pre-mortem in `docs/experiments.md` before each run; watch every GPU job to the end; check the
  loop_steer GPU queues before a launch.
- **Highlight limits up front** (ToC): report worst-behavior recall and what the monitor still misses.
