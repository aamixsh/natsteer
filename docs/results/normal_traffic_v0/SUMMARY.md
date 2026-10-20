# normal_traffic_v0: a probe that knows the model's normal traffic (2026-10-20)

Question: the writer-trained probe misses gpt-oss-20b's own misbehavior in agentic rollouts (onpolicy_v0, result
3). Can the model's normal traffic, which is free and plentiful in deployment, fix part of that? Two designs:
a two-level monitor (anomaly detector, then the probe) and a probe trained with normal traffic (whitening or
extra negatives). CPU only, existing features; layer 15; writer probe from Claude seeds (`seedsrc/claude_s4`).

| File | Content | Script |
|---|---|---|
| `anomaly_gate.json` | Anomaly AUROC (overall, per behavior), gate table, anomaly-score histograms | `scripts/anomaly_gate_check.py` |
| `anomaly_drivers.json` | What the anomaly score tracks; size of the probe direction in normal variance | `scripts/anomaly_drivers.py` |
| `whiten.json` | Plain, negatives, whitened (clean / unlabeled pool) probes over 3 scenario splits | `scripts/whiten_robust.py` |

C for the whitened probe (1e-4) was chosen on Bloom val (`scripts/whiten_check.py`).

## Setup

- **Normal traffic:** layer-15 window features of gpt-oss-20b's own on-policy rollouts (`onpolicy_direct`) from
  50 of the 100 scenarios. *Clean:* aligned turns only. *Unlabeled:* all turns (11-14% of windows misaligned).
- **Test:** own rollouts from the other 50 scenarios (66-95 misaligned conversations, about 1,300 aligned turns
  per split), and Bloom test. Indicators convention: each misaligned conversation is a positive, each aligned
  turn a negative.
- **Anomaly detector:** Mahalanobis distance (Ledoit-Wolf) on the top 64 principal components of turn-mean
  features, fit on clean normal traffic (own rollouts) or on Bloom val aligned turns (Bloom test).
- **Whitening:** x -> Sigma^-1/2 (x - mu), with mu and Sigma (Ledoit-Wolf, 20,000 windows) from normal traffic,
  then the same balanced logistic regression on the writer data. In the original space this is a probe whose L2
  penalty is w' Sigma w: using a direction costs more the more normal traffic varies along it.

## Results

**1. Misaligned turns are not anomalous.** Anomaly AUROC, misaligned vs aligned turns: Bloom test 0.50 (per
behavior: sycophancy 0.29, sandbagging 0.32, sabotage 0.35, deception 0.50, self-preservation 0.59); own rollouts
0.65 overall but at most 0.50 within each behavior. The score mostly tracks turn length (short turns are the
outliers; Spearman -0.66 on own rollouts, -0.24 on Bloom), not tool use (AUROC 0.53 / 0.44).

**2. The misalignment direction is quiet.** The probe direction holds 0.05% (own) / 0.08% (Bloom) of the
variance of normal traffic (a random direction: 0.035%); 94-95% of it lies outside the top 64 components. Misaligned
turns shift 1.56 SD along it on Bloom and 0.22 SD on own rollouts; along the top component, 0.02-0.04 SD.

**3. An anomaly gate before the probe lowers recall** (detection at 1% / 0.1% false alarms; the probe threshold
is set so that total false alarms stay at the budget):

| Gate passes | Bloom test | Own rollouts |
|---|---|---|
| all turns (no gate) | 0.46 / 0.27 | 0.02 / 0.00 |
| 50% of aligned turns | 0.36 / 0.19 | 0.02 / 0.00 |
| 20% | 0.18 / 0.09 | 0.02 / 0.00 |
| 10% | 0.04 / 0.01 | 0.06 / 0.00 |

**4. Normal traffic in training helps on own rollouts, at no cost on Bloom** (mean of 3 splits; SD in brackets):

| Probe | Own AUROC | Own detection at 1% | Own detection at 5% | Bloom test AUROC / at 1% |
|---|---|---|---|---|
| Plain (C 3e-4) | 0.745 (0.014) | 0.01 (0.00) | 0.07 | 0.948 / 0.46 |
| Normal traffic as negatives (C 3e-4) | 0.840 (0.007) | 0.09 (0.02) | 0.29 | 0.944 / 0.48 |
| Whitened, clean pool (C 1e-4) | 0.846 (0.030) | 0.11 (0.04) | 0.35 | 0.949 / 0.48 |
| Whitened, unlabeled pool | 0.820 (0.020) | 0.05 (0.01) | 0.23 | 0.950 / 0.46 |

## What it shows

- Rare in frequency is not rare in activation space: misaligned turns sit inside normal traffic and differ along
  a direction with little normal variance. A novelty score is dominated by the loud directions (length, topic) and
  cannot see it, so a novelty gate only removes positives.
- The useful part of the two-level idea is the model of normal traffic itself: used to tell the probe which
  directions are normal variation (whitening) or as negatives, it lifts own-rollout AUROC by 0.10. Whitening
  needs no labels and keeps about half the gain at 1% even when the pool is 11-14% misaligned.
- The order should be reversed for a second level: the probe screens for recall, a costlier check (follow-up
  question read, judge) on flagged steps adds precision.

Limits: one layer, one pilot; own positives are about 93% sycophancy; normal traffic and its test set come from the
same scenario writer and simulators; detection at 1% is still low (0.11), so agentic writer data is still needed;
whitening plus negatives together and other anomaly detectors were not tried.
