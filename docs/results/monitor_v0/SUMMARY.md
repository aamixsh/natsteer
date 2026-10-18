# monitor_v0: self-generation loop on gpt-oss-20b (2026-10-18)

Monitors are window probes at layer 15 (32-token window means, logistic regression, C = 3e-4 chosen on Bloom val),
turn score = max window, conversation score = max turn. Headline test: Bloom test split, rollouts of
GLM-4.7-Flash and Qwen3.6-35B-A3B, re-read by gpt-oss-20b (230 misaligned conversations, 2,659 aligned turns).
All numbers in `report.json` (made by `scripts/loop_report.py`); figure `docs/figures/loop_rounds.png`.

## M0: tools

| Check | Result |
|---|---|
| Writer refusals, 200 prompts (40 per behavior) | 0 with anti-refusal, 0 without |
| Parsed conversations | 92% with anti-refusal, 71% without (failures are format: reasoning blocks without a reply, or one turn) |
| Accepted misaligned conversations | 108 with, 100 without |
| Judge (unsteered gpt-oss-20b, low effort) vs indicators labels, 300 transcripts | recall 0.81, false-positive rate 0.00, turn agreement 0.88 (medium effort: 0.82 / 0.00 / 0.89) |
| Re-read, chunked vs full pass | equal except under 1% of tokens (MoE rounding, not at chunk boundaries) |
| Style vectors (strength 0.1 x median norm, generated tokens) | parse rate 78% vs 94% base; move re-read features 1.1-1.4 base SD along the vector |

## M1-M2: arms, 5 rounds x 500 writer prompts (Bloom test)

| Training data | Conversations | AUROC [95% CI] | TPR at 1% FPR [CI] |
|---|---|---|---|
| Prompt only, round 0 | 503 | 0.914 [0.897, 0.931] | 0.40 [0.32, 0.47] |
| Prompt only, rounds 0-4 | 2,537 | 0.918 [0.901, 0.935] | 0.32 [0.22, 0.44] |
| Anti-refusal, round 0 (seeds arm) | 559 | 0.945 [0.932, 0.957] | 0.47 [0.40, 0.55] |
| Anti-refusal + seed amplification, rounds 0-4 | 3,151 | 0.935 [0.921, 0.949] | 0.46 [0.38, 0.53] |
| Anti-refusal, round 0 (full arm) | 552 | 0.949 [0.937, 0.961] | 0.37 [0.31, 0.49] |
| Anti-refusal + seeds + style vectors, rounds 0-4 | 2,840 | 0.939 [0.925, 0.953] | 0.43 [0.36, 0.51] |
| External writer (indicators, Claude Sonnet) | 500 | 0.904 [0.885, 0.922] | 0.26 [0.17, 0.35] |
| External writer | 2,000 | 0.908 [0.894, 0.923] | 0.24 [0.16, 0.30] |

Recall at 1% FPR by behavior (anti-refusal + seeds, rounds 0-4): sandbagging 0.88 (83), sycophancy 0.42 (53),
sabotage 0.18 (45), deception 0.07 (29), self-preservation 0.05 (20). Every arm misses deception and
self-preservation at this threshold.

## What it shows

1. Self-written data transfers to other models' real rollouts: AUROC 0.935-0.949 vs 0.908 for Claude-written
   transcripts at equal or larger size, and about double the detection at 1% FPR (0.43-0.47 vs 0.24). The
   external data wins on its own writer (indicators eval 0.986 vs 0.955-0.966).
2. Anti-refusal raises transfer although the writer never refused: anti-refusal round 0 0.945/0.949 vs prompt
   only 0.914 (intervals barely overlap). Its role here is not to unblock writing; why the text differs is open.
3. Amplification does not help yet: no arm improves from round 0 to round 4; more rounds add data near what is
   there (median distance to the nearest Bloom positive falls only 41 -> 40).
4. Style vectors steer (1.1-1.4 SD along the vector) but do not raise detection.
5. Replicate noise is large at 1% FPR: the two identical round-0 configurations differ by 0.10 (0.47 vs 0.37).

Limits: one model, one layer, writer mode only; Bloom positives are dominated by single-turn sandbagging; the
judge misses about 19% of indicators positives; seeds come from Claude-written transcripts.
