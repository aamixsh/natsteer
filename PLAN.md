# natsteer: plan

MATS project 1 with Isak Etinger. This file is the current plan only. Rewrite it when the plan changes
(do not append history: results live in [docs/presentation.md](docs/presentation.md) and
`docs/results/`, setup in [docs/HANDOFF.md](docs/HANDOFF.md), history in git). Theory of change:
[docs/toc.md](docs/toc.md). Last updated 2026-10-18.
**Status: loop built and run on gpt-oss-20b (M0-M2, `docs/results/monitor_v0/`). Self-written data beats
Claude-written data on Bloom; amplification rounds do not help yet. The adversarial arm is on hold (section 3.3).**

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

## 2. Status (2026-10-18)

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
| 2 Verify | Aligned rewrite as matched negative; unsteered judge labels each turn, rates realism, rejects hazard | Accepted positives and negatives with turn labels |
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

1. **Why amplification does not help.** Seeds chosen by low out-of-fold score may be judge errors or odd
   writes rather than new styles; check a sample by hand. Then aim the budget at what the monitor misses
   (deception, self-preservation) and at multi-turn settings where the behavior develops late. Selection must
   never use the frozen sets.
2. **Why anti-refusal helps transfer** although nothing was refused: compare text statistics and re-read
   features of round-0 anti-refusal vs prompt-only conversations (length, reasoning share, how explicit the
   misaligned reasoning is); run 3 replicates of each round-0 config to size the effect.
3. Per-request style vectors in one vLLM batch (style rounds 30 -> 8 min).
4. Early-warning curve (detection vs window position) and the OOD benchmarks in `ood_misalignment_eval/`.
5. Review the shortened sabotage and sandbagging definitions with Isak; decide on the adversarial arm.
6. Re-cut the deck to goal → loop → checks → results; first testbed to an appendix.
7. Qwen3.6-35B-A3B (M3).

## 6. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Positive class | Assistant turns (reasoning included) that show the behavior | The monitor target |
| Generation mode | Writer: the whole conversation from a definition and few-shot seeds | Cheap breadth; the model's own phrasing; actor mode later |
| Negatives | Matched aligned rewrite (same situation) + non-misaligned turns + indicators and Bloom negatives | Probes otherwise learn the situation, not the act |
| Labels | Unsteered judge per turn; steering condition is provenance only | |
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
