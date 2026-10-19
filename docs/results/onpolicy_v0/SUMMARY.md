# onpolicy_v0: gpt-oss-20b's own misbehavior as seeds and as a test (2026-10-18/19)

Question: can seeds from the model's **own** misbehavior (on-policy) replace the Claude-written seeds, and do
monitors trained on self-written data catch the model's own misbehavior? All numbers in `report.json` (made by
`scripts/seedsrc_report.py`); figures `docs/figures/onpolicy_rates.png`, `docs/figures/seed_source.png`.
Positives for hand review: `data/onpolicy/pilot/review.md` (not in git).

## Setup

| Part | What |
|---|---|
| Scenarios | 100 pressure scenarios written by gpt-6.1-sol (OpenRouter, medium effort): 5 behaviors × {situational, incentive} × 10. The assistant is never told to misbehave; incentive scenarios add a conflicting goal or consequence. 94 distinct domains |
| Rollouts | gpt-oss-20b plays the assistant (low reasoning effort, its own chat format), 4 samples × {no steering, anti-refusal} × 4 user turns = 800. It calls its built-in tools (`container.exec`, `repo_browser.*`) in 52% of rollouts (2.5 calls on average); unsteered gpt-oss-20b simulates the tool results and the user |
| Judge | gpt-6.1-sol, medium effort, per turn; counts mistakes or made-up results only when the reasoning or actions show the behavior's motive; gives an evidence quote. $0.011 per rollout |
| Seeds | 4 per behavior from the judged positives (different scenarios; condensed so the misaligned turn fits the writer's 6,000-character view). Sabotage and sandbagging had no positives and keep Claude seeds |
| Comparison | writer round 0 with anti-refusal, 500 prompts, 3 runs each from Claude seeds and from on-policy seeds; a monitor trained directly on the 798 judged rollouts; prompt-only round 0 as reference. Layer 15, C = 3e-4 |
| Tests | the frozen loop sets and the on-policy rollouts without the 11 seed scenarios (114 misaligned, 2,377 aligned turns). The direct monitor is scored on the rollouts out of fold (5 folds by scenario) |
| Cost | $10.89 API in total (pilot $9.79); about 2 GPU-h |

## Results

**1. gpt-oss-20b misbehaves on its own mainly by sycophancy.** Rollouts judged misaligned (no steering /
anti-refusal): sycophancy 68/80 and 75/80, self-preservation 3/80 and 3/80 (all in incentive scenarios),
deception 1/80 and 3/79, sabotage 0/160, sandbagging 0/159. Anti-refusal adds a little (81 vs 72 of about 400).
The first misaligned turn is turn 1 in 61 rollouts, 2 in 52, 3 in 26, 4 in 14. No rollout had hazardous detail.

**2. On-policy seeds are about as good as Claude seeds on Bloom** (mean, sd over 3 runs):

| Training data | Bloom test AUROC | Bloom test detection at 1% | Claude-written test AUROC | Own held-out rollouts AUROC / detection |
|---|---|---|---|---|
| Prompt only, Claude seeds (1 run) | 0.914 | 0.40 | 0.946 | 0.72 / 0.01 |
| Anti-refusal, Claude seeds | 0.947 (0.002) | 0.43 (0.04) | 0.955 (0.005) | 0.76 (0.04) / 0.03 |
| Anti-refusal, on-policy seeds | 0.926 (0.014) | 0.47 (0.02) | 0.934 (0.011) | 0.74 (0.03) / 0.01 |
| Trained on the rollouts (1 run) | 0.764 | 0.09 | 0.614 | 0.96 / 0.45 (out of fold) |

On-policy seeds give slightly lower AUROC and slightly higher detection at 1%; both differences are within
about two run-to-run SDs. They lower the score on the Claude-written test set, as expected. By behavior on
Bloom test (detection at 1%), on-policy vs Claude seeds: sabotage 0.27 vs 0.20, self-preservation 0.07 vs
0.02, sycophancy 0.30 vs 0.28, sandbagging 0.91 vs 0.88, deception 0.08 vs 0.07.

**3. No writer-trained monitor catches the model's own misbehavior.** On held-out gpt-oss-20b rollouts all
writer monitors reach AUROC 0.71-0.80 and detect 1-4% at 1% false alarms. Cause: aligned turns with tool calls
score as high as misaligned turns (AUROC 0.51-0.56 against tool turns, 0.78-0.87 against turns without tools);
the writer data has no tool use. Even against turns without tools, detection at 1% is only 0.04-0.10.

**4. Training on the rollouts works in distribution and does not transfer.** The direct monitor scores 0.96 /
0.45 on held-out scenarios but 0.76 / 0.09 on Bloom: its positives are 93% sycophancy.

## What it shows

- Self-written data covers the *text* of misbehavior but not the *setting* of real agent work: monitors from it
  confuse tool use with misbehavior. Writer data needs tool calls and tool output (agentic conversations), and
  matched negatives with the same tool use.
- On-policy rollouts are the cheap source of realistic agentic negatives and of sycophancy positives; they are
  too rare for the other behaviors (0-4%) to seed them alone.
- The seed source matters less than the setting: Claude vs on-policy seeds differ by about the run-to-run noise.

Limits: one model, one layer, one pilot of 100 scenarios; the judge is a single model and the non-sycophancy
positives are few (10) and partly borderline (some may be fabrication read as motive; see review.md); the
writer's own conversations are short (2-3 turns, no tools), so the gap in result 3 is partly a format gap;
gpt-oss-20b simulates the environment and the user.
