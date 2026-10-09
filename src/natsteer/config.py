"""Paths, model ids and fixed constants shared across experiments."""

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "natsteer"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Persistent outputs and job temp space. Defaults are the repo's `data/` and `scratch/`
# entries (symlinks to large disks on the lab server); override per machine with
# NATSTEER_DATA and NATSTEER_SCRATCH.
DATA_ROOT = Path(os.environ.get("NATSTEER_DATA", REPO_ROOT / "data"))
RUNS_DIR = DATA_ROOT / "runs"
MODELS_DIR = DATA_ROOT / "models"
CHECKPOINTS_DIR = DATA_ROOT / "checkpoints"
SCRATCH_ROOT = Path(os.environ.get("NATSTEER_SCRATCH", REPO_ROOT / "scratch"))

MODEL_ID = "openai/gpt-oss-20b"
N_LAYERS = 24
D_MODEL = 2880

SAE_REPO = "andyrdt/saes-gpt-oss-20b"
SAE_LAYERS = (3, 7, 11, 15, 19, 23)
# trainer_0 is k=64, trainer_1 is k=128 (see each trainer's config.json).
SAE_TRAINERS = {64: 0, 128: 1}
GOODFIRE_SAE_REPO = "Goodfire/Hackathon-gpt-oss-20b-SAE-l15"
GOODFIRE_SAE_FILE = "topk_sae_l15_exp16_k32.pt"

# The stock chat template stamps today's date into the system message, which
# changes activations from day to day. Every rendering uses this date instead.
FIXED_DATE = "2025-08-05"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(name: str) -> Path:
    """Persistent run directory: DATA_ROOT/runs/<name>/<UTC stamp>."""
    path = RUNS_DIR / name / utc_stamp()
    path.mkdir(parents=True, exist_ok=False)
    return path


def scratch_dir(tag: str) -> Path:
    """Job temp dir following the lab rule <UTC stamp>--<project>--<unique>."""
    path = SCRATCH_ROOT / f"{utc_stamp()}--{PROJECT}--{tag}-{secrets.token_hex(2)}"
    path.mkdir(parents=True, exist_ok=False)
    return path
