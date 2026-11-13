# natsteer

Natural synthetic data for active learning of activation classifiers (MATS project 1).
The plan is in [PLAN.md](PLAN.md).

## Layout

```
src/natsteer/
  config.py      paths (data/runs, scratch naming), model and SAE ids, fixed chat date
  harmony.py     gpt-oss harmony rendering with a segment label per token; streaming SegmentTracker
  model.py       load gpt-oss-20b (bf16 or mxfp4), resid_post capture, segment-aware Steerer, batched sampling
  sae.py         andyrdt BatchTopK SAEs and the Goodfire layer-15 TopK SAE, reconstruction metrics
  features.py    batched pooled-feature extraction from clean forward passes
  probes.py      pooling (user_mean, prompt_last, cot_mean, final_mean), logistic probes, TPR at fixed FPR
  laser.py       LASER active learning (uncertainty band plus farthest-point), uncertainty-only and random arms
  grader.py      policy grader: gpt-oss-safeguard-20b on vLLM, policy file in, 0/1 labels out
  vllm_utils.py  vLLM engine with Triton attention (FlashInfer fails on this GPU)
  vllm_steer.py  activation addition inside vLLM (all-position, prompt-only, optional per-token reconstruction cap)
  attribution.py SAE-latent delta-attribution on paired behavior / no-behavior completions
  probe_eval.py  probe grid with clustered bootstrap intervals, precision/recall, matched discriminator
  refusal.py     refusal testbed: prompt sets, labels from the model's own answer, quality and diversity metrics
  data.py        loaders: XSTest, Aegis 2.0, hh-rlhf red-team, HarmfulQA, WildChat-1M, sycophancy evals
  convo.py       multi-turn conversations: schema, writer text format, indicators and Bloom loaders, rendering
  prompts.py     every prompt text: definitions, scope clause, settings, writer, branch, rewrite, judge, on-policy
                 scenario, user and tool simulators (scripts/show_prompts.py prints them with filled examples)
  behaviors.py   the five behavior cards (definitions from prompts.py, data maps)
  writer.py      writer (optionally agentic: tool steps), aligned-branch, rewrite and judge prompt builders and parsers
  convfeats.py   clean re-read of conversations into 32-token window and turn features
  monitor.py     window probes, indicators evaluation convention, out-of-fold scores
  amplify.py     next-round seeds (missed and far) and per-cluster style vectors
  onpolicy.py    pressure scenarios, the model as the assistant (its chat format, its own tool calls), simulated
                 user and tools, on-policy judge prompt builders
  remote.py      judge calls (gpt-6.1-sol via the OpenAI API, or OpenRouter) with a spend ledger and a hard cap
  qwen.py        Qwen3.8-27B chat rendering (effort levels, native tool calls) and reply parsing
  mathgrade.py   MATH answer checking for the checkable sandbagging test
  apikeys.py     API keys read from ~/.bashrc as text, never exported or printed
scripts/
  check_machine.sh, setup_machine.sh, prefetch.py,    new-machine setup (docs/HANDOFF.md): check, install, download,
  pull_data.sh, claude_session.sh, verify_machine.sh  copy results, carry Claude sessions, verify
  check_vllm_steering.py, compare_e0.py               verification helpers (vLLM vs HF steering, e0 reference)
  make_presentation_figures.py                        figures for docs/presentation.md
  e0_smoke.py, e0_noise_floor.py, e0_mxfp4_check.py   infrastructure checks (data/runs/e0_*)
  build_intent_pool.py, build_aegis_pool.py           pools under data/pools/
  extract_features.py                                 <rows>.jsonl -> <rows>.feats.npz
  grade.py, validate_grader.py                        policy labels and agreement with human labels
  laser_baseline.py                                   LASER vs random vs uncertainty curves (data/runs/laser_baseline)
  syco_pipeline.py                                    sycophancy proxy end to end, staged
  refusal_pipeline.py, run_refusal_overnight.sh       refusal testbed (first pass): steering, attribution, synthetic data, probes
  steer_v1.py, steer_v1_extra.py, morning_summary.py   second pass: wider latent search, sycophancy testbed, LASER warm start, reachability
  steer_v2.py, morning_summary_v2.py                   third pass: wider latent search, prompt-matched positives, probe controls, layer sweep
  steer_e3.py                                          E2 at scale and E3 (capped per-token steering, best-of-K selection)
  run_after.sh                                         run a pipeline after a running chain finishes
  loop_data.py, loop_generate.py, loop_features.py,    self-generation loop: frozen data, write/rewrite/judge (vLLM),
  loop_round.py, run_loop.sh, loop_external.py,        re-read (HF), train/evaluate/amplify, one arm for R rounds,
  loop_judge_check.py, loop_report.py                  external-writer baseline, judge check, report (monitor_v0)
  onpolicy_pilot.py, onpolicy_seeds.py,               on-policy pilot: scenarios, rollouts, remote judge; seeds and
  run_round0.sh, seedsrc_report.py                    training conversations; one round 0 from a seed file; report (onpolicy_v0)
  anomaly_gate_check.py, anomaly_drivers.py,          normal traffic and the probe (normal_traffic_v0): anomaly gate,
  whiten_check.py, whiten_robust.py                   what the anomaly score tracks, whitened probes
  style_sweep.sh, style_sweep_report.py               style-vector strength sweep
  queue_branch_rerun.sh, branch_rerun_report.py      branch negatives + gpt-6.1-sol judge rerun, queued on free GPUs (branch_sol_v0)
  tool_balance_check.py                               tool-use shortcut: fixes at training time (CPU)
  show_prompts.py                                     every prompt, seed and on-policy scenario in one file for review
  judge_ensemble.py                                   other judges (Qwen3.6-35B-A3B, Qwen3.8-27B, gpt-oss-20b), 3 samples each
  judge_audit.py                                      judge agreement and the monitors under each judge's labels
  curate_seeds.py                                     round-0 seeds with hand-checked ids per behavior (configs/seeds_curated.json)
  probe_variants.py                                   probe variants on the same data: multiple-instance, whitened variance, per behavior, layers
  checklist_check.py                                  checklist judge (one yes/no per requirement) against the hand checks
  eval_audit.py, learning_curve.py, tool_balance_turns.py  evaluation audit, learning curve, tool use by turn label (CPU)
  sandbag_math.py                                     checkable sandbagging: MATH under an evaluation threat
  spawn_safe.py                                       starts a Qwen vLLM job safely (use for every Qwen vLLM run)
  refusal_direction.py                                Qwen3.8-27B anti-refusal direction: candidates, screen, test
  persona_neurons.py                                  persona-neuron steering (n3838 at layer 34) and its judged sweep
  persona_vectors.py                                  per-behavior persona vectors (2507.21509): materials, extraction,
                                                      vectors (--reread-system), steering sweep
  qwen_report.py                                      Qwen monitors (default: writer + own rollouts, span labels, scenario folds; E19-E21)
  steered_train.py                                    on-policy rollouts as monitor training data (E11; span arms E20, CPU)
  selfpres_probe.py                                   probes trained on steered self-preservation rollouts (E14, E16, E17, CPU)
  span_labels.py                                      window span labels from the judge's evidence quotes (E17, CPU)
  portable_onpolicy.py                                standalone on-policy runner for another machine (Kilian handoff)
  transcript_viewer.py                                local HTML viewer of every arm's rollouts per scenario (data/viewer/)
configs/policies/   grader policies (toy_programming.md is a plumbing test only)
tests/              CPU tests
docs/results/  small run reports behind the deck (committed); see docs/results/README.md
docs/experiments.md  experiment log: question, pre-mortem before the run, result and verdict after
data/      outputs (pools, features, runs); not in git
scratch/   job temp files; not in git
```

## Setup and runs

New machine (vast.ai, RunPod, lab server): [docs/HANDOFF.md](docs/HANDOFF.md). In every shell: `source env.sh`
(shared caches, `uv` on PATH). Outputs go to `data/`, scratch dirs to `scratch/` (`NATSTEER_DATA`, `NATSTEER_SCRATCH`).

```bash
.venv/bin/python -m pytest -q tests
```

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/e0_smoke.py --keep-model
```

```bash
GPU=1 ./scripts/run_refusal_overnight.sh          # refusal testbed, about 3 h; --smoke for a tiny check run
GPU=1 PIPELINE=steer_v1.py ./scripts/run_refusal_overnight.sh   # second pass, about 6 h (steer_v1_extra.py: 1 h more)
GPU=1 PIPELINE=steer_v2.py ./scripts/run_refusal_overnight.sh   # third pass, about 7 h
GPU=1 PIPELINE=steer_e3.py ./scripts/run_refusal_overnight.sh   # E2 at scale and E3, about 1.5 h
```

Self-generation loop (needs `data/external/misalignment-indicators`, cloned from GitHub):

```bash
.venv/bin/python scripts/loop_data.py
```

```bash
GPU=0 scripts/run_loop.sh full 5 100
```

On-policy rollouts and seeds (judge key in `~/.bashrc`, read only by `natsteer.apikeys`; spend capped by `--cap`).
Qwen steered arms add `--pv <behavior>@<layer>@<coef>[@cot][@from<N>]` (`cot`: steer the reasoning only; `from<N>`:
turns before N unsteered; recipes `deception@32@4@cot@from2` (E7) and `self_preservation@32@6@cot@from2` (E12); `--seed`
for another sample) and start through `scripts/spawn_safe.py`:

```bash
GPU=0 .venv/bin/python scripts/onpolicy_pilot.py --out data/onpolicy/pilot --cap 60
```

```bash
.venv/bin/python scripts/onpolicy_seeds.py && cp data/onpolicy/pilot/seeds_onpolicy.jsonl data/loop/seeds_onpolicy.jsonl
```

```bash
GPU=0 scripts/run_round0.sh data/loop/arms/seedsrc/onpolicy_s2 data/loop/seeds_onpolicy.jsonl 2
```

Current results: [docs/presentation.md](docs/presentation.md) and `docs/results/`; plan and next todo: [PLAN.md](PLAN.md).

## Conventions

- Every rendering goes through `harmony.render`, which fixes the system-message date
  (`config.FIXED_DATE`); the stock template stamps today's date and shifts activations.
- `load_model("bf16")` dequantizes the MXFP4 experts (about 39 GB on GPU). It is exact
  and differentiable, so it is the default for attribution, steering and GCG.
- SAE metrics drop token position 0 (huge norm, excluded from SAE training) and report
  segments separately; the system message was mostly excluded from SAE training.
- Steering masks come from `SegmentTracker`, so "CoT only" means the same tokens in
  teacher-forced passes and in cached generation.
- The harmful-intent grader policy goes in `configs/policies/harmful_intent.md`; it is the
  label definition and is written by the researchers, not generated.
- Activations depend slightly on batch composition: the bf16 MoE runs one matmul per
  expert, sized by its routed tokens, so rerouting some tokens changes rounding for others.
