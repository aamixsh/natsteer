# Committed result files

Small run reports, copied from `data/runs/` (not in git) so the deck and its figures can be
rebuilt from a clone: `.venv/bin/python scripts/make_presentation_figures.py` reads these.
Generated text, activations and features stay in `data/` and are not committed.

## refusal_v0 (2026-10-10, run `data/runs/refusal_pipeline/20261010T154724Z`)

Refusal testbed, `scripts/refusal_pipeline.py`, one GPU, 2 h 46 min.

| File | Content |
|---|---|
| `directions_info.json` | Difference-of-means direction sizes, norms and cosines per layer |
| `sweep_table.json` | vLLM steering sweep: direction x layer x positions x coefficient, plus random controls (192 samples each) |
| `sweep_hf_table.json` | HF steering with CoT-only and final-only masks (32 samples each) |
| `attribution.json` | Delta-attribution of SAE latents (layers 15 and 19) on same-prompt refuse/comply pairs |
| `latents_table.json` | Steering with single attributed latents, delta-activation latents, random latents and bundles |
| `generator_choices.json` | The generators picked for synthetic data |
| `probes_report.json` | Probes on natural, synthetic and mixed data; naturalness discriminator; text diversity |
| `syco_attribution.json` | Sycophancy delta-attribution (sampling and attribution only; nothing steered yet) |
| `timings.log` | Stage durations |

## steer_v1 (2026-10-10 to 11, `data/pools/steer_v1`)

Second pass, `scripts/steer_v1.py` and `steer_v1_extra.py`, one GPU, 7.2 h. `SUMMARY.md` is the readable
summary; `*.json.gz` are the full probe reports and screening tables (gzip).

| File | Content |
|---|---|
| `SUMMARY.md`, `timings.log` | Tables for every stage; stage durations |
| `attribution_wide.json`, `latents_screen_table.json.gz`, `latents_validated_table.json` | Refusal: 1,500-pair attribution, 288-condition screen, validation on fresh prompts |
| `probes_report.json.gz` | Refusal probes: 10 seeds, intervals, controls, coverage, diversity |
| `syco_*` | Sycophancy: attribution, sweep, latent screen and validation, probes |
| `laser_warmstart.json`, `probes_wildchat.json` | LASER with synthetic warm start; probes on held-out WildChat refusals |
| `reach_table.json`, `timing_table.json` | Reachability (reconstruction error, nearest natural neighbor); steering timing |

## steer_v2 (2026-10-11, `data/pools/steer_v2`)

Third pass, `scripts/steer_v2.py`, one GPU, 7.1 h. `SUMMARY.md` is the readable summary; `*.json.gz` are the full
reports and screening tables (gzip).

| File | Content |
|---|---|
| `SUMMARY.md`, `timings.log` | Tables for every stage; stage durations |
| `screen2_table.json.gz`, `latents_validated2_table.json`, `generator_choices_v2.json` | Refusal: 501-condition wider screen (ranks 33 to 128, strengths 8 and 12), validation (23 latents), generator choices |
| `probes2_report.json.gz` | Refusal probes with the many-latent, multi-strength and best-bundle generators |
| `probes2_extra_report.json.gz` | Union, naturalness-filtered and multi-layer arms |
| `probes_matched_report.json.gz` | Probes on held-out WildChat refusals with prompt-matched positives |
| `probes_layers_report.json.gz` | Layer sweep (7, 11, 15, 19, 23, all concatenated) |
| `syco_*` | Sycophancy: wider screen (400 conditions), validation (39 latents), probes with a random-direction arm, CoT-length matching and budgets to 2,048 |

## steer_e3 (2026-10-11, `data/pools/steer_e3`)

E2 at scale and E3, `scripts/steer_e3.py`, one GPU, 1.3 h. `SUMMARY.md` is the readable summary.

| File | Content |
|---|---|
| `SUMMARY.md`, `timings.log` | Tables for every stage; stage durations |
| `e3_floor.json` | Natural per-token SAE reconstruction error quantiles (layers 11, 15, 19): the E3a cap |
| `reach2_table.json` | E2: distance from natural activations for every validated refusal and sycophancy condition, with yield |
| `e2_usefulness.json` | E2: per refusal latent, probe quality of its own data against yield and distance; correlations |
| `e3_alpha.json`, `e3_select_stats.json` | E3a mean per-token strength under the cap; generation yields and selection statistics |
| `e3_probes_report.json.gz` | E3 probes: harmless-prompt arms on the refusal test set, matched-prompt arms on held-out WildChat |

## monitor_v0 (2026-10-18, `data/loop/`)

Self-generation loop on gpt-oss-20b: M0 tool checks, three generation arms × 5 rounds and the external-writer
baseline, tested on Bloom rollouts. `scripts/loop_data.py`, `run_loop.sh`, `loop_external.py`, `loop_report.py`;
two GPUs, about 3 h. `SUMMARY.md` is the readable summary, `report.json` all numbers.

## onpolicy_v0 (2026-10-18/19, `data/onpolicy/pilot`, `data/loop/arms/seedsrc`)

gpt-oss-20b's own misbehavior in 100 pressure scenarios (800 rollouts, judge gpt-6.1-sol), used as writer seeds
(3 runs vs 3 with Claude seeds), as training data, and as a test set. `scripts/onpolicy_pilot.py`,
`onpolicy_seeds.py`, `run_round0.sh`, `seedsrc_report.py`; two GPUs, about 2 h, $10.89 API. `SUMMARY.md` is the
readable summary, `report.json` all numbers.

## normal_traffic_v0 (2026-10-20, CPU, existing features)

Can gpt-oss-20b's normal traffic improve the writer-trained probe on its own rollouts? An anomaly gate before the
probe (fails: misaligned turns are not anomalous) vs whitening by normal traffic or normal traffic as negatives
(AUROC 0.75 -> 0.85, no cost on Bloom), over 3 scenario splits. `scripts/anomaly_gate_check.py`,
`anomaly_drivers.py`, `whiten_check.py`, `whiten_robust.py`. `SUMMARY.md` is the readable summary.

## branch_sol_v0 (2026-10-20/21, `data/loop/arms/{branch_sol,rewrite_sol}`)

Same writes as the three Claude-seed round-0 runs, relabeled by gpt-6.1-sol, with rewrite or branch negatives
(3 arms × 3 runs): better labels raise own-rollout AUROC 0.76 -> 0.87 (0.91 with branches and whitening), Bloom about
the same. `scripts/queue_branch_rerun.sh`, `branch_rerun_report.py`; $15.1 API, about 1 GPU-h. `SUMMARY.md` is the
readable summary, `report.json` all numbers.

## judge_audit_v0 (2026-10-22/23, `data/judges/`)

Do other judges agree with gpt-6.1-sol, and do the monitor results hold under their labels? Qwen3.6-35B-A3B,
Qwen3.8-27B and gpt-oss-20b (medium effort), 3 samples each, on the 798 own rollouts, the 1,339 branch_sol writes and
their 797 branches. `scripts/judge_ensemble.py`, `judge_audit.py`; about 13 GPU-h, no API. `report.json`: agreement
(positive rates, kappa, self-consistency) and the result-6 monitors on own rollouts under each judge's labels.

## seedfix_v0 (2026-10-23/24, `data/loop/arms/seedfix_*`)

Sabotage and sandbagging writer tests: new seeds (`configs/seeds_curated*.json`, `scripts/curate_seeds.py`), writer
and judge rules v1-v3 (`natsteer.prompts`), 50 prompts per behavior per run, every accepted write hand-checked.
`summary.json`: pipeline counts, hand-check fits and failures per criterion, seed copying; $2.96 API.

## checklist_check_v0 (2026-10-24, `data/loop/arms/checklist_check.jsonl`)

The checklist judge (one yes/no per requirement, gpt-6.1-sol) against the 144 hand-checked seedfix writes.
`scripts/checklist_check.py`, $1.21 API. `report.json`: precision and recall per behavior, agreement per item.

## probe_variants_v0 (2026-10-24, CPU; 8-token features: one GPU, 77 min)

Probe variants on the branch_sol training data (3 runs): multiple-instance relabeling and a pooled turn loss,
whitened mean and variance directions, per-behavior probes, joined layers; 32- and 8-token windows
(`loop_features.py --window 8 --layers 15 --tag w8`). `scripts/probe_variants.py`. `report.json` (32 tokens) and
`report_w8.json`: mean, SD and per-run values on own rollouts (two label sets), Bloom test and val, Claude-written,
own writes.

## eval_audit_v0 (2026-10-27/28, CPU)

Evaluation audit of the gpt-oss writer monitors (branch_sol s2-s4, layer 15): AUROC and detection at each test's own 1%
with group-bootstrap intervals, detection at a threshold frozen on Bloom val, conversation-level false alarms, scoring
by misaligned turns only. `scripts/eval_audit.py`; `tool_balance.json` (tool use by turn label before and after the
branch fix) from `scripts/tool_balance_turns.py`.

## learning_curve_v0 (2026-10-28, CPU)

More or more varied writer data: monitors trained on growing random subsets of three pools (scale_v1 rounds; all
gpt-6.1-sol-labeled arms; plus the local-judge rounds), scored on Bloom test, own rollouts and checkable sandbagging.
Flat from 250 to 3,334 conversations. `scripts/learning_curve.py`; `report_pre_scale.json` is the run before scale_v1.

## qwen_port_v0 to v2 (2026-10-28 to 2026-11-12, `data/qwen/`)

Qwen3.8-27B monitors (3 writer runs, layers 24/32/40/48) on Bloom val and test, Qwen's own rollouts, the checkable
sandbagging test and every steered on-policy arm (persona vectors E1-E15, single seeds and pooled) as test rows.
`scripts/qwen_report.py`; v1 is overwritten by each report (latest: report v11, 2026-11-09, `data/qwen/report_v11.log`).
v0 used an earlier, broken anti-refusal direction (runs kept in `data/qwen/old_v0dir`). v2 `report_span.json` (E19): the
same report at layer 32 with the writer runs span-labeled (`--labels span`; log `data/qwen/report_v12_span.log`). v2
`report.json` (v13, E21, the reference): the default monitor, writer span + own unsteered span, rows on the pilot
scenarios scored on held-out scenario folds (log `data/qwen/report_v13.log`).

## steered_train_v0 to v2 (2026-11-06 to 2026-11-12, CPU; E11, E20)

Steered deception rollouts as monitor training data: writer data alone, + Qwen's unsteered rollouts, + the steered
rollouts, + both, split by scenario (3 seeds x 2 folds x 3 writer runs). v0 uses E7, v1 E7 + E10. Steered data gives
no gain on unsteered tests, so steered sets stay test-only; unsteered own rollouts are the useful addition. v2 (E20):
span arms (`wspan`, `wspan+own`, `wspan+ownspan`, same splits): own unsteered rollouts span-labeled lift held-out own
frozen detection 0.32 -> 0.66 at no Bloom cost; turn-labeled they cost Bloom 0.947 -> 0.913. `scripts/steered_train.py`.

## selfpres_probe_v0 to v5 (2026-11-08 to 2026-11-12, CPU; E14, E16, E17, E18)

Is self-preservation linearly readable? Probes trained on steered self-preservation rollouts (steered turns only),
tested on held-out scenarios of the same runs and on Bloom (including against same-topic aligned conversations), next to
the writer monitors. v0 E12, v1 E12 + E13, v2 E12 + E13 + E15 (layer 32), v3 the same at layers 24/40/48 and layer 32
reasoning-only or reply-only windows. v4 (E17) other training labels on v2's data: span labels from the judge's evidence
quotes (`scripts/span_labels.py`), top-25% windows, pooled turn loss, turn means; span labels reach 0.80 within topic.
v5 (E18) span labels on the writer runs too, alone and with the steered spans; Bloom per-behavior AUROC and frozen detection.
`scripts/selfpres_probe.py`.
