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
