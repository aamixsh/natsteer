# Moving natsteer to another machine

Code goes through git. Large outputs (`data/`) are not in git; copy them or rebuild them.
Models and datasets are downloaded fresh on the new machine.

## 1. Code

```bash
git clone <repo-url> natsteer && cd natsteer
```

## 2. Environment and downloads

```bash
./scripts/setup_machine.sh
```

This creates `data/` and `scratch/`, builds `.venv` from `uv.lock`, downloads about 40 GB
from Hugging Face (`scripts/prefetch.py`), runs the CPU tests and lists the GPUs.

Requirements:
- `uv` on PATH. Python 3.13 is fetched by uv.
- NVIDIA driver with CUDA 13 support (driver 580 or newer). The lock pins torch 2.13 (cu130) and vLLM 0.31.
- A Hugging Face login if you will use gated data later: `.venv/bin/hf auth login`.
- About 60 GB of disk: 40 GB of models, 5 GB of features, 8 GB for `.venv`.

Large disks: put data and temp files wherever the machine has room, then either symlink
`data/` and `scratch/` to them or export `NATSTEER_DATA` and `NATSTEER_SCRATCH`.

## 3. Results and features

Copy from the lab server (`data/` is a symlink there; `-L` follows it):

```bash
rsync -avPL lab-server:natsteer/data/pools lab-server:natsteer/data/runs  natsteer/data/
```

| Path | Size | If not copied |
|---|---|---|
| `data/runs/` | 0.5 MB | Run reports behind the deck and PLAN.md. **Copy**; not reproducible bit for bit. |
| `data/pools/*/*.jsonl` | 55 MB | Rebuild: `build_intent_pool.py`, `build_aegis_pool.py` (deterministic, seconds). Sycophancy natural and steered answers were sampled, so **copy** `syco_v0/*.jsonl`. |
| `data/pools/*/*.feats.npz` | 4.6 GB | Rebuild with `extract_features.py` (about 10 min on one GPU) and `syco_pipeline.py --stage features`. |
| `scratch/` | 3.5 GB | Not needed. |

To rebuild everything instead of copying, follow the appendix of `docs/presentation.md`.

## 4. Check the machine

```bash
.venv/bin/python scripts/e0_smoke.py --keep-model
.venv/bin/python scripts/e0_noise_floor.py
```

Compare with `data/runs/e0_smoke/20261008T183341Z/report.json`: SAE variance explained on CoT
should be about 0.74 (layer 15, k = 64), and the steering mask should hit exactly 690 CoT tokens.
Activation-level numbers can differ by about 2% on another GPU (see the noise-floor slide).

## Machine-specific settings

| Setting | Default | When to change |
|---|---|---|
| `NATSTEER_DATA` | `<repo>/data` | data lives elsewhere |
| `NATSTEER_SCRATCH` | `<repo>/scratch` | temp space lives elsewhere |
| `NATSTEER_VLLM_ATTENTION` | `TRITON_ATTN` | FlashInfer failed on the RTX PRO 6000 (sm_120); on H100/A100 try `FLASH_ATTN` or `FLASHINFER` for speed |
| `CUDA_VISIBLE_DEVICES` | all | scripts use device `cuda:0` |
| `HF_HOME` / `HF_HUB_CACHE` | `~/.cache/huggingface` | put the 40 GB cache on a large disk |

## Not in the repo

- The harmful-intent grading policy (`configs/policies/harmful_intent.md`) does not exist yet.
- No secrets are stored in the repo. Log in to Hugging Face separately on each machine.
