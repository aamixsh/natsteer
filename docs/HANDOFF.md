# Setting up a new machine (vast.ai, RunPod, lab server)

Point a fresh Claude Code session (or a person) at this file: "set up this machine following
docs/HANDOFF.md". It covers only moving the project to a new server. For the project itself read
[PLAN.md](../PLAN.md) (current plan and next todo) and [presentation.md](presentation.md) (results).

Everything needed is in git or downloadable. Nothing secret is stored in the repo; the Hugging Face
token stays on the machine.

## What the project needs from a machine

| Need | Detail |
|---|---|
| GPU | NVIDIA, driver 580 or newer (the locked torch 2.13 and vLLM 0.31 are CUDA 13 builds). bf16 gpt-oss-20b needs about 42 GB; vLLM runs at 85% of one GPU. Developed on RTX PRO 6000 (sm_120, 95 GiB) |
| Disk | about 120 GB: models and datasets 70 GB (grader included), `.venv` 8 GB, `data/` 10 GB, headroom. Put them on the persistent volume |
| Network | Hugging Face, PyPI, GitHub (raw files) |
| Tools | `curl`, `git`; `rsync` for copying data; `tmux` for long runs |
| Python | uv fetches Python 3.14 (the lock resolves `requires-python >=3.13` to it); nothing to install by hand |

## Steps

Run from the repo root. Each script is safe to rerun.

1. **Clone and look.**
   ```bash
   git clone git@github.com:aamixsh/natsteer.git && cd natsteer
   source env.sh && ./scripts/check_machine.sh
   ```
   Fix any FAIL. A WARN for a busy GPU means choose another GPU index below. `env.sh` puts the shared caches
   (Hugging Face models and token, uv wheels, uv Pythons) in `/workspace/cache` when `/workspace` exists
   (the persistent volume on vast.ai and RunPod), else in `.cache/shared`. Set `NATSTEER_CACHE` before
   sourcing to put them elsewhere. Compiler caches stay in `.cache/`.

2. **Hugging Face login** (ask the user for a token; never write it into the repo):
   `HF_TOKEN=...` for one shell, or `.venv/bin/hf auth login` after step 3. Checks and downloads read
   `$HF_HOME/token`.

3. **Environment and downloads** (about 40 GB with the grader):
   ```bash
   ./scripts/setup_machine.sh                 # add --skip-grader to save 13 GB, --no-prefetch to skip downloads
   ```
   It checks the machine, installs uv if missing, runs `uv sync --locked`, downloads the models, SAEs,
   datasets and the refusal prompt sets (public GitHub repo, pinned commit), rebuilds the prompt pools
   (`data/pools/{intent_v0,aegis_v0}/*.jsonl`, deterministic) and runs the CPU tests.

4. **Results and sampled data** from the old machine (skip on a fresh start; everything is rebuildable
   except sampled data and reports):
   ```bash
   ./scripts/pull_data.sh user@old-host:/path/to/natsteer         # add --big for the 8 GB of pooled features
   ```
   | Path | Size | Without it |
   |---|---|---|
   | `data/runs/` | under 1 MB | the reports behind the deck are lost; the key ones are also in `docs/results/` |
   | `data/pools/syco_v0/*.jsonl`, `data/pools/refusal_v0/*.jsonl` | 80 MB | sampled generations; rerun the pipelines (refusal about 3 GPU-hours) |
   | `data/pools/*/*.feats.npz` | 8 GB | rebuild: `extract_features.py`, `syco_pipeline.py --stage features`, `refusal_pipeline.py --stage features` |
   | `data/pools/{intent,aegis}_v0/*.jsonl` | 55 MB | rebuilt by step 3 |
   | `scratch/` | any | never needed |

5. **Verify** (about 10 minutes, one free GPU):
   ```bash
   GPU=0 ./scripts/verify_machine.sh
   ```
   Runs the tests, `e0_smoke.py --keep-model` compared with the first machine's numbers
   (`docs/results/e0_reference.json`: SAE variance explained on CoT 0.74 at layer 15 and 0.83 at layer 19,
   exactly 690 CoT-masked steering tokens, 37,807 CoT tokens, zero-vector logit difference 0), and
   `check_vllm_steering.py` (vLLM hook steering against the HF `Steerer`). Activation-level numbers can
   differ by about 2% on another GPU (MoE routing flips), hence the tolerances. Look for `VERIFY PASSED`.

6. **Optional: carry Claude Code conversations over.** On the old machine
   `./scripts/claude_session.sh export` (writes `data/handoff/claude_session/`, copied by step 4), on the new
   one `./scripts/claude_session.sh import`, then `claude --resume <session id>` from the repo root.
   Sessions can contain file contents; keep them out of git.

7. **Run.** Long jobs under `tmux` or `setsid nohup`; pick the GPU explicitly:
   ```bash
   GPU=1 ./scripts/run_refusal_overnight.sh          # refusal testbed, resumable, about 3 h on one RTX PRO 6000
   ```
   Outputs go to `data/pools/refusal_v0/` and `data/runs/refusal_pipeline/<stamp>/`.

## Machine-specific settings

| Setting | Default | Change when |
|---|---|---|
| `NATSTEER_CACHE` | `/workspace/cache` if `/workspace` exists, else `.cache/shared` | the volume is mounted elsewhere |
| `NATSTEER_DATA`, `NATSTEER_SCRATCH` | `<repo>/data`, `<repo>/scratch` | bulk storage lives elsewhere (symlinks also work) |
| `NATSTEER_BIN` | `/workspace/bin` if `/workspace` exists, else `.cache/bin` | uv should live elsewhere |
| `NATSTEER_VLLM_ATTENTION` | `TRITON_ATTN` | FlashInfer fails on sm_120; on H100 or A100 try `FLASH_ATTN` or `FLASHINFER` for speed |
| `GPU` / `CUDA_VISIBLE_DEVICES` | none (scripts require `GPU=`) | always choose a free GPU; scripts use the first visible device |
| `IDLE` (overnight runner) | 900 s | a stage legitimately goes quiet for longer |

## Gotchas

- **Non-interactive shells (nohup, cron, agent tools) do not read `~/.bashrc`.** Always `source env.sh`;
  otherwise caches silently fall back to the home directory and models download again.
- **vLLM on sm_120 crashes with FlashInfer**; `natsteer.vllm_utils.make_llm` forces Triton attention.
- **vLLM hooks** need `enforce_eager=True`, prefix caching off and `VLLM_ALLOW_INSECURE_SERIALIZATION=1`;
  `natsteer.vllm_steer.make_steerable_llm` sets them. vLLM generates about 6,500 tok/s here against about
  120 tok/s for HF steered generation.
- **A vLLM engine once hung at start-up** (waiting on a lock before loading weights). The overnight runner
  kills a stage with no output for `IDLE` seconds (default 900) and retries it up to three times.
- **Model repos ship duplicate `original/` and `metal/` weights** (13 GB each); `prefetch.py` skips them.
- **Activations reproduce only to about 2%** between identical computations (MoE routing flips), so exact
  comparisons across machines or batch compositions fail; compare against the noise floor.
- `pkill -f <pattern>` can match the invoking shell; kill by PID. Do not recreate `.venv` while a job runs.
- The harmful-intent grading policy (`configs/policies/harmful_intent.md`) does not exist; the researchers
  write it, and only the optional intent track needs it.

## Checklist for the agent doing the setup

Ask the user, then proceed: (1) the Hugging Face token or whether the machine is already logged in,
(2) the old machine to pull results from, if any, (3) which GPU to use and whether another job is running
on the others, (4) whether to download the 13 GB grader. After step 5 passes, report the GPU, driver,
free disk and the verify output, then read PLAN.md and continue with its next todo.
