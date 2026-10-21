# natsteer: plan

MATS project 1 with Isak Etinger. This file is the current plan only. Rewrite it when the plan changes
(do not append history: results live in [docs/presentation.md](docs/presentation.md) and
`docs/results/`, setup in [docs/HANDOFF.md](docs/HANDOFF.md), history in git). Theory of change:
[docs/toc.md](docs/toc.md). Last updated 2026-10-21.
**Status: loop built and run on gpt-oss-20b (M0-M2, `docs/results/monitor_v0/`). Self-written data beats
Claude-written data on Bloom; amplification rounds do not help yet. On-policy pilot (`docs/results/onpolicy_v0/`):
gpt-oss-20b misbehaves on its own mostly by sycophancy; on-policy seeds match Claude seeds. Normal traffic
(`docs/results/normal_traffic_v0/`): an anomaly gate hurts; whitening the probe by the model's normal traffic helps.
**Labels matter most** (`docs/results/branch_sol_v0/`): with gpt-6.1-sol as judge instead of the local model, the
same writes give AUROC 0.87 instead of 0.76 on the model's own rollouts (Bloom 0.94, about the same); branch
negatives add a little (0.88) and whitening more (0.91 / 0.21 detection at 1%). The earlier "tool turns look like
misbehavior" was mostly label noise. gpt-6.1-sol (realism >= 2) and branches are now the defaults. Agentic writer
data is built (smoke: 68% parse, 97% of those with tool steps). Style vectors are on hold. Next: a full agentic
round. The adversarial arm is on hold (section 3.3).**

## 1. Goal

Make **activation monitors** of misalignment reliable by generating **diverse, natural misaligned
conversations** that do not exist in natural data. A monitor is a linear probe on the residual stream of a
deployed open-weight reasoning model that reads every assistant token and flags a conversation trending
toward a misaligned behavior (threat model in the ToC: an agent under reward pressure that deceives about its
task, sabotages work, resists shutdown or replacement). Monitors fail where their training data has no
coverage, so the data must cover the *class*, not a few examples.

Method: **the aligned model writes the data itself.** It gets a behavior definition plus a few example
conversations (few-shot) and writes *new flavors*: new setting, new user, new way of going wrong, with the
assistant's private reasoning. Two uses of steering: (1) **anti-refusal**, in case the model refuses or sanitizes
the request; (2) **style vectors** found from the generated data that push the writer toward distinct styles of a
behavior. The loop is **self-amplifying**: each round's accepted conversations that the current monitor misses
become the next round's seeds and style vectors, so the data spreads over the class and the monitor gets
stronger each round. Naturalness and diversity under control are the research problem.

Success: on frozen test sets the loop never reads, a monitor trained on R rounds of loop data beats, at equal
budget, monitors trained on (a) round-0 data without amplification, (b) the same prompts without steering,
(c) transcripts written by an external model (misalignment-indicators, Claude Sonnet). Models: gpt-oss-20b, then
Qwen3.6-35B-A3B. End state: one command from a behavior description to a monitor.

## 2. Status (2026-10-21)

- **Built and smoke-tested (gpt-oss-20b):**
  - `scripts/loop_data.py` → `data/loop/`: round-0 seeds (4 per behavior from the indicators training data,
    hazard domains excluded), a disjoint held-out seed set, the external-writer pool (half of each indicators
    file by id hash) and frozen eval sets: `eval/indicators.jsonl` (other half), `eval/bloom_val.jsonl`
    (477 conversations, 144 misaligned), `eval/bloom_test.jsonl` (793, 230 misaligned; downloaded from
    `kzhou35/misalignment-indicators-bloom-rollouts`). Bloom labels are the Opus per-turn ground truth; ambiguous
    and audit-suspicious rollouts are dropped, as in the indicators repo.
  - `natsteer.convo`: conversation schema, the tagged text format the writer uses, loaders, multi-turn harmony
    rendering with a segment and a turn index per token (Bloom tool results stay inside their turn).
  - `natsteer.behaviors`: five behavior cards with Bloom's definitions (sabotage and sandbagging shortened and
    without security flaws; **needs researcher review**), a scope clause (no hazardous detail), indicator and
    Bloom folder maps.
  - `natsteer.vllm_steer`: anti-refusal = directional ablation at every layer (ported from loop_steer; direction
    `v4_baseline` L15 copied to `data/checkpoints/anti_refusal_gptoss20b.pt`; loop_steer measured refusal 92% →
    2.1%); new `positions="gen"` (steer generated tokens only).
  - `scripts/loop_generate.py`, `natsteer.writer`: writer → aligned rewrite of the assistant turns (matched
    negative: same system and user turns) → unsteered judge (per-turn labels, realism 1–5, hazard check).
    Accept a positive if some turn shows the behavior, realism ≥ 3, no hazard; a negative if no turn does.
  - `scripts/loop_features.py`, `natsteer.convfeats`: clean re-read with HF bf16, mean residual over 32-token
    windows (cut at reasoning/reply changes) and per turn, layers 11/15/19; long Bloom transcripts run in 2,048-token
    chunks with a KV cache (agrees with a full pass except under 1% of tokens, the MoE rounding effect). Speed
    4.5k–7k tokens/s.
- **Also built:** `natsteer.monitor` (window probe, indicators evaluation convention, group bootstrap),
  `natsteer.amplify` (missed-and-far seed selection, per-cluster style vectors with SAE latents),
  `scripts/loop_round.py`, `scripts/run_loop.sh` (one arm, R rounds), `scripts/loop_external.py`,
  `scripts/loop_judge_check.py`, `scripts/loop_report.py`; CPU tests in `tests/test_loop.py`.
- **Results (`docs/results/monitor_v0/SUMMARY.md`, Bloom test, layer 15, C = 3e-4 chosen on Bloom val):**
  - Writer refusals 0/200 with and without anti-refusal; anti-refusal raises parse rate (92% vs 71%), not yield.
  - Judge vs indicators labels: recall 0.81, false-positive rate 0.00.
  - Self-written data: AUROC 0.935-0.949, TPR at 1% FPR 0.37-0.47; Claude-written (indicators) at 500-2,000
    conversations: 0.904-0.908, 0.24-0.26. Prompt-only (no steering): 0.914-0.918, 0.32-0.40.
  - No arm improves over rounds 0-4 (seed amplification, style vectors). Style vectors do steer (1.1-1.4 SD).
  - Every arm misses deception and self-preservation at 1% FPR (recall under 0.15); sandbagging 0.8-0.95.
  - Two identical round-0 runs differ by 0.10 in TPR at 1% FPR: use replicates before claiming differences.
- **On-policy pilot (`docs/results/onpolicy_v0/`, $10.89 API):** `natsteer.onpolicy`, `natsteer.remote`
  (OpenRouter, spend ledger with a hard cap), `natsteer.apikeys` (keys read from `~/.bashrc` as text, never
  exported), `scripts/onpolicy_pilot.py`, `onpolicy_seeds.py`, `run_round0.sh`, `seedsrc_report.py`.
  - 100 scenarios (gpt-6.1-sol) × 4 samples × {none, anti-refusal}, 4 turns; gpt-oss-20b calls its built-in tools
    (simulated by unsteered gpt-oss-20b) in 52% of rollouts. Judge gpt-6.1-sol (medium), motive required.
  - Misaligned: sycophancy 89%, self-preservation 4% (incentive only), deception 3%, sabotage and sandbagging 0%.
  - Writer round 0 from on-policy seeds vs Claude seeds (3 runs each), Bloom test: AUROC 0.926 vs 0.947, detection
    at 1% 0.47 vs 0.43 (within noise).
  - On held-out own rollouts every writer monitor fails (AUROC 0.71-0.80, detection at 1% 0.01-0.04): aligned
    tool-use turns score like misbehavior. A monitor trained on the rollouts gets 0.96 / 0.45 there but 0.76 on
    Bloom (its positives are 93% sycophancy).
- **2026-10-20 (in the deck):**
  - **Branch negatives** (`loop_generate.py --negatives branch`, now the default): keep the conversation up to the
    first misaligned turn, write an aligned replacement, then one turn at a time a new user message (the original
    one as a hint) and an aligned reply. Smoke test: 25/26 branches parse (21 full, 4 short), 23 accepted. The
    local judge is the weak point (a late branch point; a half-sycophantic "aligned" reply accepted). No monitor
    has been trained on branch negatives yet (next todo 1).
  - **Style strength sweep** (`--style-scale`, `scripts/style_sweep.sh`, `style_sweep_report.py`; one vector per
    behavior, every prompt styled, 150 prompts per scale): stronger does not work in writer mode. Format OK /
    positives per prompt / realism: none 0.94 / 0.56 / 3.65; x1 (alpha 0.1, current) 0.68 / 0.33 / 3.31 (shift
    1.64 SD, spread unchanged); x2 0.13 / 0.03 / 3.00; x4 0 / 0 (incoherent text). The stored strength is already
    about 1.5x the natural misaligned-aligned gap. The vector comes from the assistant role but steers the writer.
  - **Normal-traffic model** (`scripts/anomaly_gate_check.py`, `anomaly_drivers.py`, `whiten_check.py`,
    `whiten_robust.py`; layer 15, writer probe from Claude seeds; normal = aligned turns of own rollouts from half
    the scenarios): misaligned turns are not anomalous (Mahalanobis AUROC 0.50 on Bloom test, at most 0.5 within
    each behavior on own rollouts). The anomaly score mostly tracks turn length (short turns are outliers; Spearman
    -0.66 on own rollouts); the probe direction holds 0.05-0.08% of normal variance and 94-95% of it lies outside
    the top 64 PCs. An anomaly gate before the probe lowers Bloom-test detection at 1% from 0.46 to 0.36 / 0.18 /
    0.04 (gate passes 50 / 20 / 10% of aligned turns). Whitening the probe by normal-traffic covariance (C 1e-4,
    3 scenario splits; `docs/results/normal_traffic_v0/`): own held-out 0.846 / 0.11 (plain 0.745 / 0.01), Bloom
    test unchanged (0.949 / 0.48); from an unlabeled pool with 11-14% misaligned windows 0.820 / 0.05. Normal
    traffic as negatives: 0.840 / 0.09 (about equal; needs clean traffic).
- **2026-10-21: branch rerun with gpt-6.1-sol (`docs/results/branch_sol_v0/`, $15.1).** `loop_generate.py --judge remote`
  (gpt-6.1-sol via `natsteer.remote`, spend ledger `data/loop/arms/sol_spend.jsonl` with a hard cap, `--judge-only`,
  `--pos-judgments`; vLLM loads only when a stage needs it), `scripts/queue_branch_rerun.sh` (API prep, then one
  worker per GPU that waits for the GPU to be free), `scripts/branch_rerun_report.py`. Same writes as the three
  Claude-seed round-0 runs; 3 arms × 3 runs. Mean AUROC / detection at 1%, Bloom test | own rollouts | own splits
  plain -> whitened: rewrites + local judge 0.947 / 0.43 | 0.764 / 0.03 | 0.734 -> 0.842; rewrites + gpt-6.1-sol
  0.937 / 0.44 | 0.869 / 0.16 | 0.843 -> 0.883; branches + gpt-6.1-sol 0.938 / 0.43 | 0.878 / 0.13 | 0.876 / 0.14 ->
  0.907 / 0.21. The judges agree on 596 positives; gpt-6.1-sol drops 215 of the local judge's 811 (sabotage 191 ->
  101, sandbagging 128 -> 81) and adds 268; it flags 16% of the local judge's aligned rewrites. Misaligned vs aligned
  tool turns on own rollouts: 0.55 -> 0.77-0.81 (vs 0.89-0.91 without tools). gpt-6.1-sol rates realism lower, so
  the bar is >= 2. Caveat: the own-rollout labels come from gpt-6.1-sol too.
- **2026-10-21: agentic writer data, built.** `--agentic-frac`: the writer prompt asks for an agent with tools
  (`[TOOL CALL]` / `[TOOL RESULT]` steps inside its turn, stored like the on-policy rollouts: a call is an assistant
  message starting with `[tool call]`, a result a user-role message with `tool: true`); branches keep the tool use
  (the original first misaligned turn is shown when it used tools); reasoning is optional right after a tool result.
  `tests/test_agentic.py`. Smoke (`data/loop/smoke/agentic`, 50 prompts, all agentic): 68% parse after the
  reasoning rule (42% before), 97% of those with tool steps; gpt-6.1-sol keeps 11 of 21 positives at first parse.
  Unchanged for conversations without tools, except 64 of 13,179 stored writes that pasted tool output into a reply.
- **Speed:** a base round takes about 8 min (writes 500, rewrites, judge, re-read, train); style rounds about
  30 min because each style vector runs as its own small batch (fix: per-request vectors in one batch).
- **First testbed (closed; `docs/results/steer_v1`, `steer_v2`, `steer_e3`):** tool validation and three lessons:
  input distribution dominates probe transfer; steering variety alone does not help; reasoning-segment steering
  moves gpt-oss where answer steering does not.
- **GPUs:** other projects' jobs are capped at 32 GB per GPU, so both GPUs are usable: vLLM at
  `gpu_memory_utilization` 0.55 on one, the HF re-read (peak about 55 GB) on the other. `loop_generate.py` kills
  its vLLM engine process on exit (an orphaned engine held 55 GB once).

## 3. Design

### 3.1 Behaviors

Five, matching the Bloom evaluation behaviors: **strategic deception**, **covert sabotage**, **strategic
sandbagging**, **self-preservation**, **sycophancy**. Harmful compliance (CBRN, cyber, weapons) is not a target.

### 3.2 The loop (per model; behaviors run together)

| Step | What happens | Output |
|---|---|---|
| 1 Write | Definition + k = 2 seeds (weighted) + sampled setting (deployment × domain, 2–3 turns); base steering (anti-refusal or none); a share of prompts adds a style vector on generated tokens | Candidate conversations with provenance |
| 2 Verify | Unsteered judge labels each turn, rates realism, rejects hazard; matched negative = aligned branch from the first misaligned turn (replacement turn, then new user turns and aligned replies), judged the same way | Accepted positives and negatives with turn labels |
| 3 Re-read | Unsteered HF forward pass on the exact text | Window and turn features |
| 4 Train | Logistic regression on window features of all accepted data so far; turn score = max window, conversation score = max turn | Monitor P_{r+1} |
| 5 Amplify | **Seeds:** accepted positives with low out-of-fold monitor score, spread by farthest-point sampling in activation space, become new seeds (old seeds downweighted). **Style vectors:** cluster accepted positives in activation space; per cluster, mean of positive turns minus mean of their matched negatives → style vector, with nearest SAE latents on its card | S_{r+1}, V_{r+1} |
| 6 Measure | P_{r+1} on the frozen sets; yield, refusal, realism; coverage and number of styles | Round report |

### 3.3 Arms (same per-round budget)

| Arm | Steering | Amplification | Tests |
|---|---|---|---|
| `prompt` | none | none (S_0 every round) | Plain self-generation |
| `seeds` | anti-refusal | seeds only | Does monitor-guided seed choice help |
| `full` | anti-refusal + style vectors | seeds + style vectors | Do style vectors add coverage |
| `external` | — | — | Indicators transcripts at the same budget |

Later: random-vector control; actor mode (the model *is* the assistant, steered).
**On hold: adversarial arm** (write while steering against the current monitor's direction, keep what the judge
accepts). It is the part of the plan closest to monitor evasion; a safety classifier stopped work while it was
being built. It stays out of the code until the user and Isak decide whether and how to run it.

### 3.4 Evaluation (frozen; the loop never reads it)

| Level | Set | Question |
|---|---|---|
| 1 | Own generations from the held-out seeds | Does the data work at all |
| 2 | Indicators eval half (Claude-written) | Transfer across writers |
| 3 | Bloom val (dev: layer, regularization) and Bloom test (headline): GLM-4.7-Flash and Qwen3.6 rollouts | Transfer to real rollouts |

Convention of the indicators paper: each misaligned conversation is one positive (max over turns); each turn
without the behavior is one negative. Report AUROC, TPR at 1% and 5% turn-level FPR, per-behavior TPR,
bootstrap intervals over conversation groups. Later: MASK, DeceptionBench, Sycophancy-Eval, Agentic Misalignment
from `ood_misalignment_eval/`; early-warning curve (detection vs window position in the turn).

## 4. Experiments

- **M0 Tools (about 2 GPU-h).** Writer with and without anti-refusal, 40 prompts per behavior: refusal, parse rate,
  yield, realism. Judge agreement with indicators labels (positives, hard negatives, benign over-triggering) at
  low and medium effort. Style-vector strength sweep (parse rate and yield) once vectors exist.
- **M1 Round 0 (about 1 GPU-h).** 100 prompts per behavior; first monitor; external arm; levels 1–3. Re-read of
  the eval sets once (Bloom val + test about 15M tokens, about 1 h).
- **M2 The loop (about 25 min per round per arm; 5 rounds × 3 arms, two GPUs in parallel, about 4 h).** Detection,
  coverage and style count per round. This is the core claim.
- **M3 Qwen3.6-35B-A3B (download 70 GB).** Repeat; compare with the released indicators probes (AUROC 0.90).
- **M4 Automation.** One command from a behavior description to seeds, vectors, rounds, monitor and report.

## 5. Next todo (pick up here)

1. **Agentic writer data, round 2: balance tool use between the classes.** Round 1 (`arms/agentic_sol`, 2026-10-21,
   $13.6) made the tool confusion worse (misaligned vs aligned tool turns 0.657 vs 0.770 without agentic data): 87%
   of the misaligned turns have tool steps vs 51% of the aligned branch turns. Next: (a) benign agentic writes (the
   writer writes aligned agent conversations with heavy tool use) as extra negatives; (b) branch steps told to keep
   acting through tools as much as the original; (c) a CPU check first: the model's own aligned tool turns from the
   fit half as negatives, and turn weights that equalize the tool share. Also fix replies without an `[ASSISTANT]`
   tag (half of the remaining branch failures) and add a writer rule against misbehavior instructed by the system
   prompt ("nobody asks for or approves the misaligned choice"). Ledger: $28.9 of $50.
2. **Monitors that know normal traffic.** First three CPU checks: whitening and negatives together; normal traffic
   from a different source than the test set (now both come from the same scenario writer); the Rachmil et al.
   whitened anomaly score (2512.03994) on our data, to locate the disagreement with result 6 of
   normal_traffic_v0. Then whitened probes as the default, hard-negative mining from a large pool of the model's
   own benign traffic, and detection at 0.1% false alarms (Alpaca is too small and off-distribution for that,
   2507.12691). Then a short write-up: whitened probe + failed anomaly gate.
3. **Judge audit.** gpt-6.1-sol is now the default judge. Hand-check a sample where the two judges disagree
   (especially sabotage and sandbagging), the on-policy positives (`data/onpolicy/pilot/review.md`) and branches.
   The own-rollout labels also come from gpt-6.1-sol: check part of the gain is not judge agreement (a second
   judge or hand labels on a sample of own rollouts).
4. **Two-level monitor:** the probe screens every token (high-recall threshold); flagged steps get a follow-up
   question read by a probe (FAR.AI 2610.12445: "Did you do anything the user did not ask you to do?", prefilled
   "Yes."/"No." contrast) or a judge, possibly async. The cascade itself is standard (2601.04603, 2601.11516,
   2506.10805); measure precision at 0.1% false alarms. Test the follow-up read offline first (re-read with the
   question after each turn); add a multi-layer ensemble (11/15/19 extracted).
5. **Text-based steering (not started) and rare behaviors on-policy:** in the on-policy runner, insert a steering
   sentence into the model's own reasoning (analysis channel) and let it continue: the simple baseline to
   activation steering (mask the inserted tokens in training, paraphrase widely, five behaviors only). Also harder
   incentive scenarios and longer agentic tasks. Compare with natural rollouts.
6. **Style / behavior vectors: on hold.** The current ones are k-means clusters of re-read writer text minus
   negatives that differ in content, at a fixed layer and an unchecked strength, so they carry topic and format.
   When resumed, rebuild them by the persona-vector recipe (2507.21509, `safety-research/persona_vectors`): per
   behavior (or a named style of it), 5 elicit/suppress system-prompt pairs and 40 questions (20 extract, 20 eval)
   from a strong model; gpt-oss answers as the assistant, 10 samples per question and prompt; judge trait 0-100
   and coherence; keep pairs with trait >= 50 / < 50 and coherence >= 50; mean over response tokens (reasoning
   and final separately) at every layer; vector = raw difference of means. Choose layer and coefficient by
   steering the eval questions without a system prompt (judged trait up, coherence >= 75). Steer generated
   tokens in actor mode (the on-policy runner), not the writer (x1 is the writer's ceiling, x2 breaks it).
7. Speed of the rollout runner (one process per steering config, continuous batching); why anti-refusal helps
   transfer; early-warning curve and OOD benchmarks; tool calls rendered in gpt-oss's own format (commentary
   channel, `to=` recipient) instead of folded text, for the own rollouts and the agentic writes alike.
8. With Isak: the sabotage and sandbagging definitions (gpt-6.1-sol finds no behavior in most of these writes:
   no evaluation or stake is set up); decide on the adversarial arm; Qwen3.6-35B-A3B (M3).

**Later (from 2604.22167, tail-risk estimation with steered proposals):** (a) estimate the natural rate of rare
behaviors from steered rollouts with importance weights; (b) score how on-policy a steered rollout is by its
likelihood ratio p_unsteered / p_steered (the re-read gives p_unsteered; vLLM logprobs give p_steered) and filter or
weight steered data by it; (c) steer only up to a switch point, then let the model continue unsteered.

## 6. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Positive class | Assistant turns (reasoning included) that show the behavior | The monitor target |
| Generation mode | Writer: the whole conversation from a definition and few-shot seeds | Cheap breadth; the model's own phrasing; actor mode later |
| Negatives | Matched aligned branch (same prefix, regenerated from the first misaligned turn) + non-misaligned turns + the model's normal traffic (whitening, negatives) | Probes otherwise learn the situation, not the act; false alarms come from normal traffic the probe never saw |
| Labels | gpt-6.1-sol per turn (realism >= 2); steering condition is provenance only | The local judge's labels limited transfer to the model's own rollouts (branch_sol_v0) |
| Activations | Recomputed without steering on unmodified weights | The monitor must match deployment |
| Diversity | Monitor-missed seeds + data-derived style vectors + sampled settings | Variety alone did not help in the first testbed |
| Probe | Window means (32 tokens), max over windows per turn, layer 15 first | Fits early warning; comparable with the indicators protocol |
| Metric | TPR at 1% turn-level FPR on Bloom test; AUROC | Comparable with the indicators paper |
| Content scope | Portrayals of misaligned assistants; no hazardous detail; transcripts stay in `data/` | Monitors need the behavior, not hazardous knowledge |

## 7. Findings that constrain the design

- First testbed: topic dominates probe transfer; reasoning-segment steering works on gpt-oss; recomputation noise
  about 2% from MoE routing; distance from normal does not predict useful data.
- loop_steer: single-direction ablation at every layer works on gpt-oss-20b; activation addition is weaker.
- Literature: token probes on Sonnet-written transcripts reach AUROC 0.90 on Qwen3.6 out of distribution
  (2606.24251); off-policy data fails for text-ambiguous behaviors (2511.17408); probes often read the situation,
  not the action (2606.30449); persona vectors are automated from descriptions (2507.21509); steering generated
  tokens only works best (2605.31328); SAE feature names need causal checks (2601.05679).
- FAR.AI 2610.12445: probes trained mostly off-policy (163k rows, 28 families) reach AUROC about 0.98 on on-policy
  agentic sabotage (SHADE-Arena, Qwen3.5-397B); about 3k-row subsets score 0.02-0.11 lower; more families help at
  equal size; small models are much worse (Qwen3.5-2B 0.59-0.78); off-policy twins keep on-policyness from being a
  shortcut; thresholds are set on held-out honest rollouts, no anomaly detection. Misalignment that comes from a
  persona may be missed by probing the default assistant (relevant to writer mode).
- Normal traffic (search of 2026-10-20): no published activation monitor whitens a supervised probe by the
  covariance of deployment traffic. Covariance-corrected probes use the training covariance (2310.06824; Apollo's
  LDA probe, 2502.03407, no gain over logistic regression); deployment statistics are used for thresholds
  (2502.03407, 2606.24251) or as an anomaly score: Rachmil et al. 2512.03994 whiten by compliant traffic and
  detect policy violations as OOD (up to 86% F1; conflicts with our gate result); anomaly detection is mixed in
  2504.08812 and evadable in 2412.09565. The mechanism is classical: CORAL (1511.05547), AdaBN (1603.04779), SpAR
  (2312.17463: uses unlabeled target spectrum against covariate shift), matched filter vs RX detector in
  hyperspectral detection. Probe-then-judge cascades are standard (2601.04603, 2601.11516, 2506.10805).

## 8. Sources

Misalignment indicators (arXiv 2606.24251; `safety-research/misalignment-indicators`, cloned to
`data/external/`; Bloom rollouts on HF); the other references as in section 7; behavior taxonomy and dataset audit in
[docs/misalignment_research_2026-10-14.md](docs/misalignment_research_2026-10-14.md).

## 9. Risks and rules

- **Content:** behaviors of an assistant only, never hazardous knowledge; the judge rejects hazardous detail;
  transcripts stay in `data/` and are never released.
- **The loop can fool itself:** seeds are chosen by the monitor and labels come from the judge; hand-audit a
  sample per round; eval sets are frozen.
- **Depiction is not enactment:** writer text may lack a real model's decision process; Bloom level 3 is the test,
  actor mode the fallback.
- **Highlight limits up front** (ToC): report worst-behavior recall and what the monitor still misses.
