# branch_sol_v0: a stronger judge and branch negatives (2026-10-20/21)

Question: every monitor result before this run was trained on rewrite negatives (assistant turns rewritten in
place, user turns kept) labeled by the local judge (gpt-oss-20b, low effort). Do branch negatives (regenerate from
the first misaligned turn on) and a stronger judge (gpt-6.1-sol, medium effort, via OpenRouter) change the results?
All numbers in `report.json` (made by `scripts/branch_rerun_report.py`); runs by `scripts/queue_branch_rerun.sh`.

## Setup

| Part | What |
|---|---|
| Writes | The 500 writes of each of the three Claude-seed round-0 runs of monitor_v0 (seeds 2, 3, 4: `arms/seeds/r0`, `arms/full/r0`, `arms/seedsrc/claude_s4/r0`), copied, so the positives' text is the same in every arm |
| rewrite_local | The monitor_v0 runs as they were: rewrites, local judge, realism >= 3 |
| rewrite_sol | The same writes and rewrites, judged by gpt-6.1-sol; realism >= 2 |
| branch_sol | The same writes judged by gpt-6.1-sol (the judgments shared with rewrite_sol); branches from its first misaligned turn, judged by gpt-6.1-sol; realism >= 2 |
| Realism bar | gpt-6.1-sol rates realism lower (mode 2 of 5): at >= 3 it keeps 138 of 449 writes in run s2 with 3 sabotage and 0 sandbagging; at >= 2, 295 with all behaviors (local judge at >= 3: 267) |
| Training, tests | Round 0, layer 15, C = 3e-4. Bloom val and test, Claude-written, own written held-out seeds, gpt-oss-20b's own rollouts without the seed scenarios, and own rollouts over 3 scenario splits plain and whitened by normal traffic (as in normal_traffic_v0) |
| Cost | $15.1 API (judging 1,339 writes, 941 rewrites, 797 branches; about $0.005 per call); about 1 GPU-h |

## Results

**1. The judges disagree on a third of the positives.** Over the 1,339 writes: the local judge keeps 811, gpt-6.1-sol
864; they agree on 596. gpt-6.1-sol drops 215 of the local judge's positives and adds 268. It finds the behavior in
far fewer sabotage (101 vs 191) and sandbagging (81 vs 128) writes, and in more deception (216 vs 145) and
sycophancy (246 vs 141) writes. Its usual reason for sandbagging: the reasoning names sandbagging but no evaluation
or self-relevant stake is set up. Of the rewrites the local judge passed as aligned, gpt-6.1-sol flags 16% as still
misaligned. Of the branches, 68-72% are kept (sycophancy branches are often still sycophantic).

**2. Better labels fix most of the miss on the model's own rollouts; Bloom stays the same** (mean of 3 runs; SD in
brackets; AUROC / detection at 1% false alarms):

| Arm | Train pos / neg | Bloom test | Own rollouts | Own, 3 splits, plain | Own, 3 splits, whitened |
|---|---|---|---|---|---|
| rewrite_local | 270 / 292 | 0.947 (0.002) / 0.43 | 0.764 (0.046) / 0.03 | 0.734 / 0.02 | 0.842 / 0.13 |
| rewrite_sol | 288 / 257 | 0.937 (0.003) / 0.44 | 0.869 (0.020) / 0.16 | 0.843 / 0.11 | 0.883 / 0.17 |
| branch_sol | 288 / 203 | 0.938 (0.003) / 0.43 | 0.878 (0.022) / 0.13 | 0.876 / 0.14 | **0.907 / 0.21** |

**3. The tool-use confusion was mostly label noise.** Misaligned conversations vs aligned turns with tool calls on own
rollouts: AUROC 0.554 (local labels), 0.806 (rewrite_sol), 0.770 (branch_sol); vs aligned turns without tools:
0.828, 0.889, 0.912. Tool turns are still the hardest negatives.

**4. Branches add a little over rewrites with the same labels:** +0.01 to +0.03 AUROC on own rollouts (plain and
whitened), the same on Bloom, with a fifth fewer negatives. Whitening still helps on top (+0.03 AUROC; detection at 1%
0.14 -> 0.21).

**5. Agentic writes make the tool confusion worse** (arm agentic_sol, run 2026-10-21: new writes with half of the
prompts agentic, branches, gpt-6.1-sol; 3 runs, $13.6): Bloom test 0.941 / 0.46, own rollouts 0.864 / 0.06, splits
0.857 -> 0.886 / 0.21 whitened; misaligned vs aligned tool turns 0.657 (branch_sol 0.770), vs turns without tools
0.928. Cause: the writer puts the misbehavior into turns where the agent acts through tools, and the aligned branches
often answer without tools. In complete pairs, 87% of misaligned turns have tool steps (1.35 calls per turn) vs 51%
of aligned branch turns (0.66). Training on complete pairs only does not help (0.642). About half of the agentic
branches also break off ("short"): one-line `[TOOL CALL] name {...}` tags (now parsed; rescues 39% of the failed
steps) and replies without an `[ASSISTANT]` tag. 62% of agentic writes parse.

**6. Fix at training time** (`scripts/tool_balance_check.py`, `tool_balance.json`; CPU; mean of 3 runs × 3 scenario
splits, test = own rollouts of the held-out half). Agentic arm: plain 0.857 / 0.10 (tool turns 0.656); window weights
that equalize the tool share 0.871 / 0.15 (0.733); plus the model's own aligned tool-turn windows from the fit half as
negatives **0.905 / 0.30 (0.893)**, Bloom 0.938 / 0.45; plus all own aligned windows 0.898 / 0.24 (0.819), Bloom
0.934 / 0.40; whitened 0.886 / 0.21 (0.829); whitened plus own aligned 0.899 / 0.24, Bloom 0.932 / 0.38. Branch arm
(no agentic writes): plus own aligned tool turns 0.903 / 0.21 (0.944); whitened 0.907 / 0.21, but Bloom detection at
1% falls 0.43 -> 0.39 (with own negatives too, 0.34). Caveat: the added negatives and the test come from the same
scenario writer and simulators.

Two checks of the fix (`tool_balance_sources.json`, same splits). **Unlabeled:** all own tool-turn windows of the fit
half, misaligned ones included (3.5-6.6% of them), give 0.898 / 0.28 (tool turns 0.884), Bloom 0.938 / 0.45; branch
arm 0.898 / 0.21 (0.941). So the fix needs no labels. **Another source:** the 57,581 aligned tool-turn windows of
Bloom val (other models' rollouts, gpt-oss-20b reading them) give 0.889 / 0.17 (tool turns 0.818) and lower Bloom test
to 0.912 / 0.41; branch arm 0.893 / 0.13 (0.923), Bloom 0.910 / 0.40. Part of the gain from own tool turns may come
from the shared source; the Bloom test drop is likely because Bloom positives have the same style as the added
negatives.

## What it shows

- Label quality was the main limit on the model's own rollouts, not the missing tool use: the local judge (the
  model judging its own writes at low effort) accepted positives without the behavior and negatives with it.
  onpolicy_v0's result 3 ("tool turns look like misbehavior") is mostly this.
- A small Bloom drop (0.947 -> 0.937, about 3 SDs) comes with the new labels; Bloom labels come from Opus with the
  Bloom definitions, and the sandbagging-heavy Bloom test may favor the local judge's looser sandbagging positives.
- gpt-6.1-sol becomes the loop's judge, with realism >= 2; branch negatives stay the default.
- Tool use must be balanced between the classes, not only present: aligned negatives need as much tool use as the
  misaligned turns (benign agentic writes, aligned branches that keep acting through tools, or the model's own
  aligned tool turns as negatives).

Limits: one round per arm, three runs; the gpt-6.1-sol judgments are one sample each (medium effort, not
deterministic); the own-rollout labels come from the same judge model (gpt-6.1-sol with the on-policy note), so part of
the gain may be judge agreement rather than truth; own positives are about 93% sycophancy.
