# Source from the repo root in every new shell:
#   source env.sh
# Machine setup (/workspace server): models, HF token, uv wheels and uv-managed Pythons live in
# the shared caches under /workspace/cache (used by all projects). Compiler caches (vLLM,
# Triton, TorchInductor) stay inside this project in .cache/. Only sets variables that are
# not already set, so ~/.bashrc or the caller can still override.

NATSTEER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NATSTEER_ROOT
: "${NATSTEER_DATA:=$NATSTEER_ROOT/data}"
: "${NATSTEER_SCRATCH:=$NATSTEER_ROOT/scratch}"
export NATSTEER_DATA NATSTEER_SCRATCH

SHARED=/workspace/cache
C="$NATSTEER_ROOT/.cache"
: "${HF_HOME:=$SHARED/hf}"                        # models, datasets, and the HF token
: "${UV_CACHE_DIR:=$SHARED/uv}"
: "${UV_PYTHON_INSTALL_DIR:=$SHARED/uv-python}"   # the Python that .venv links to
: "${PIP_CACHE_DIR:=$SHARED/pip}"
: "${VLLM_CACHE_ROOT:=$C/vllm}"
: "${TRITON_CACHE_DIR:=$C/triton}"
: "${TORCHINDUCTOR_CACHE_DIR:=$C/torchinductor}"
: "${XDG_CACHE_HOME:=$C/xdg}"
export HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR VLLM_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_ALLOC_CONF
unset C SHARED

# uv lives in /workspace/bin (the interactive ~/.bashrc adds it; non-interactive shells do not).
case ":$PATH:" in *":/workspace/bin:"*) ;; *) export PATH="/workspace/bin:$PATH" ;; esac

mkdir -p "$NATSTEER_DATA" "$NATSTEER_SCRATCH" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
