# Source from the repo root on machines without separate data/temp disks:
#   source env.sh
# Keeps every cache, temp file and the uv-managed Python inside the project directory,
# so nothing important lives under $HOME (which may not persist on cloud machines).
# Only sets variables that are not already set, so a lab profile (e.g.
# /etc/profile.d/lab-storage.sh) that ran first keeps its shared caches.

NATSTEER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NATSTEER_ROOT
: "${NATSTEER_DATA:=$NATSTEER_ROOT/data}"
: "${NATSTEER_SCRATCH:=$NATSTEER_ROOT/scratch}"
export NATSTEER_DATA NATSTEER_SCRATCH

C="$NATSTEER_ROOT/.cache"
: "${HF_HOME:=$C/huggingface}"                    # models, datasets, and the HF token
: "${UV_CACHE_DIR:=$C/uv}"
: "${UV_PYTHON_INSTALL_DIR:=$C/uv-python}"        # the Python that .venv links to
: "${VLLM_CACHE_ROOT:=$C/vllm}"
: "${TRITON_CACHE_DIR:=$C/triton}"
: "${TORCHINDUCTOR_CACHE_DIR:=$C/torchinductor}"
: "${XDG_CACHE_HOME:=$C/xdg}"
: "${TMPDIR:=$NATSTEER_SCRATCH/tmp}"
export HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR VLLM_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME TMPDIR
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_ALLOC_CONF
unset C

case ":$PATH:" in *":$NATSTEER_ROOT/.cache/bin:"*) ;; *) [ -d "$NATSTEER_ROOT/.cache/bin" ] && export PATH="$NATSTEER_ROOT/.cache/bin:$PATH" ;; esac

mkdir -p "$NATSTEER_DATA" "$NATSTEER_SCRATCH" "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR"
