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
