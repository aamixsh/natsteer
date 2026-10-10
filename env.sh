# Source from the repo root in every new shell (non-interactive shells, nohup and cron do not read ~/.bashrc):
#   source env.sh
# Where things live:
#   shared caches (HF models and token, uv wheels, uv-managed Pythons, pip): $NATSTEER_CACHE
#       default /workspace/cache when /workspace exists (vast.ai / RunPod persistent volume, shared with
#       other projects), else <repo>/.cache/shared. Set NATSTEER_CACHE to put them elsewhere.
#   compiler caches (vLLM, Triton, TorchInductor): <repo>/.cache
#   outputs: $NATSTEER_DATA (default <repo>/data); job temp dirs: $NATSTEER_SCRATCH (default <repo>/scratch)
#   uv binary: $NATSTEER_BIN (default /workspace/bin when /workspace exists, else <repo>/.cache/bin)
# Only sets variables that are not already set, so ~/.bashrc or the caller can still override.

NATSTEER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NATSTEER_ROOT
: "${NATSTEER_DATA:=$NATSTEER_ROOT/data}"
: "${NATSTEER_SCRATCH:=$NATSTEER_ROOT/scratch}"
export NATSTEER_DATA NATSTEER_SCRATCH

if [ -z "${NATSTEER_CACHE:-}" ]; then
  if [ -d /workspace ]; then NATSTEER_CACHE=/workspace/cache; else NATSTEER_CACHE="$NATSTEER_ROOT/.cache/shared"; fi
fi
if [ -z "${NATSTEER_BIN:-}" ]; then
  if [ -d /workspace ]; then NATSTEER_BIN=/workspace/bin; else NATSTEER_BIN="$NATSTEER_ROOT/.cache/bin"; fi
fi
export NATSTEER_CACHE NATSTEER_BIN

C="$NATSTEER_ROOT/.cache"
: "${HF_HOME:=$NATSTEER_CACHE/hf}"                        # models, datasets, and the HF token
: "${UV_CACHE_DIR:=$NATSTEER_CACHE/uv}"
: "${UV_PYTHON_INSTALL_DIR:=$NATSTEER_CACHE/uv-python}"   # the Python that .venv links to
: "${PIP_CACHE_DIR:=$NATSTEER_CACHE/pip}"
: "${VLLM_CACHE_ROOT:=$C/vllm}"
: "${TRITON_CACHE_DIR:=$C/triton}"
: "${TORCHINDUCTOR_CACHE_DIR:=$C/torchinductor}"
: "${XDG_CACHE_HOME:=$C/xdg}"
export HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR VLLM_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_ALLOC_CONF
unset C

case ":$PATH:" in *":$NATSTEER_BIN:"*) ;; *) export PATH="$NATSTEER_BIN:$PATH" ;; esac

mkdir -p "$NATSTEER_DATA" "$NATSTEER_SCRATCH" "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$NATSTEER_BIN"
