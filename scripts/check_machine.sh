#!/usr/bin/env bash
# Pre-flight check for a new machine: driver, GPUs, disk, network, tools, login. Read-only.
# Exit code 1 if a hard requirement fails; warnings do not fail.
#   ./scripts/check_machine.sh
cd "$(dirname "$0")/.."
source env.sh
fail=0
ok()   { echo "  ok    $*"; }
warn() { echo "  WARN  $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }

echo "== GPUs and driver"
if command -v nvidia-smi >/dev/null; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  major=${driver%%.*}
  if [ "${major:-0}" -ge 580 ]; then ok "driver $driver (CUDA 13 wheels need 580 or newer)"; else bad "driver $driver is older than 580; the locked torch/vLLM wheels are CUDA 13"; fi
  while IFS=, read -r idx name total used; do
    used=${used// MiB/}; total=${total// MiB/}
    msg="GPU $idx: $name, $((total / 1024)) GiB total, $((used / 1024)) GiB in use"
    if [ "$used" -gt 2048 ]; then warn "$msg (busy: pick another GPU with CUDA_VISIBLE_DEVICES / GPU=)"; else ok "$msg"; fi
    if [ "$total" -lt 45000 ]; then warn "GPU $idx has under 45 GB: bf16 gpt-oss-20b needs about 42 GB (the mxfp4 path needs 13 GB)"; fi
  done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits)
else
  bad "nvidia-smi not found"
fi

echo "== disk (need about 120 GB: models 40, venv 8, data 10, headroom)"
for d in "$NATSTEER_ROOT" "$NATSTEER_CACHE"; do
  mkdir -p "$d" 2>/dev/null
  free=$(df -BG --output=avail "$d" | tail -1 | tr -dc 0-9)
  if [ "${free:-0}" -ge 120 ]; then ok "$d: ${free} GB free"; else warn "$d: only ${free} GB free"; fi
done
echo "  shared cache: $NATSTEER_CACHE   outputs: $NATSTEER_DATA   scratch: $NATSTEER_SCRATCH"

echo "== tools"
for t in curl git rsync tmux; do
  if command -v $t >/dev/null; then ok "$t"; else warn "$t not found (curl and git are required; rsync for pull_data.sh; tmux for long runs)"; fi
done
if command -v uv >/dev/null; then ok "uv $(uv --version | cut -d' ' -f2)"; else warn "uv not found; setup_machine.sh installs it into $NATSTEER_BIN"; fi

echo "== network"
for u in https://huggingface.co https://pypi.org https://raw.githubusercontent.com; do
  if curl -sfI -m 10 "$u" >/dev/null; then ok "$u"; else bad "cannot reach $u"; fi
done

echo "== Hugging Face login"
if [ -n "${HF_TOKEN:-}" ] || [ -f "$HF_HOME/token" ]; then ok "token found (HF_TOKEN or $HF_HOME/token)"; else warn "no token: run '.venv/bin/hf auth login' after setup (needed for gated data and to avoid rate limits)"; fi

[ $fail -eq 0 ] && echo "machine check passed" || echo "machine check FAILED"
exit $fail
