#!/usr/bin/env bash
# Universal bundle sync — fetch every stored pre-built wheel from the
# registries into companion/vendor/archives/ so the brain can provision fully
# offline on this host.
#
# Fetches (skipping anything already present):
#   - Windows wheels : PyTorch CUDA trio (torch/torchvision/torchaudio +cu126
#                      cp313 win_amd64, from download.pytorch.org), TensorFlow
#                      2.21.0 and ONNX Runtime GPU 1.28.0 (from PyPI)
#   - Linux wheels   : the 15 nvidia cu126 runtime wheels pinned by
#                      torch 2.9.1+cu126 (from PyPI; see nvidia-cu126-linux.txt)
#
# Safe to re-run: files that already exist with the expected size are kept.
# Run on the host you want to provision (Windows fetches win wheels, Linux
# fetches the manylinux nvidia set). Use `--force` to re-download everything.
#
# Requires: curl, python 3.8+ (only for size math), bash.
set -euo pipefail

# This script lives in companion/vendor/archives/ — that IS the archive dir.
ARCHIVES="$(cd "$(dirname "$0")" && pwd)"
cd "$ARCHIVES"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

dl() {
  local name="$1" url="$2" expected="$3"
  if [ "$FORCE" -eq 0 ] && [ -f "$name" ]; then
    local have
    have=$(stat -c '%s' "$name" 2>/dev/null || echo 0)
    # Exact size match + zip integrity: a file can be the right size yet
    # corrupt (interleaved/partial downloads), so only keep verified wheels.
    if [ "$have" -eq "$expected" ] && zip_ok "$name"; then
      echo "keep:   $name"
      return
    fi
    echo "refetch: $name (existing file failed size/integrity check)"
  fi
  echo "fetch:  $name"
  if [ "$FORCE" -eq 1 ]; then rm -f "$name"; fi
  curl -sL --retry 3 -C - -o "$name" "$url"
  local now
  now=$(stat -c '%s' "$name" 2>/dev/null || echo 0)
  if [ "$now" -eq "$expected" ] && zip_ok "$name"; then
    echo "done:   $name"
  else
    echo "WARN:   $name incomplete ($now/$expected) — re-run to resume"
  fi
}

zip_ok() {
  # Cheap zip integrity probe: unzip -t is fast (~seconds for 600MB wheels)
  # and catches truncation even when the size coincidentally matches.
  if command -v unzip >/dev/null 2>&1; then
    unzip -tqq "$1" >/dev/null 2>&1
  else
    python -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile('$1').testzip() is None else 1)" 2>/dev/null
  fi
}

# ---- Windows wheels (PyTorch CUDA 12.6 trio) ----------------------------
dl "torch-2.9.1+cu126-cp313-cp313-win_amd64.whl" \
  "https://download.pytorch.org/whl/cu126/torch-2.9.1%2Bcu126-cp313-cp313-win_amd64.whl" \
  2584515620
dl "torchvision-0.24.1+cu126-cp313-cp313-win_amd64.whl" \
  "https://download.pytorch.org/whl/cu126/torchvision-0.24.1%2Bcu126-cp313-cp313-win_amd64.whl" \
  8808733
dl "torchaudio-2.9.1+cu126-cp313-cp313-win_amd64.whl" \
  "https://download.pytorch.org/whl/cu126/torchaudio-2.9.1%2Bcu126-cp313-cp313-win_amd64.whl" \
  2050333

# ---- Windows wheels (PyPI) ----------------------------------------------
dl "tensorflow-2.21.0-cp313-cp313-win_amd64.whl" \
  "https://files.pythonhosted.org/packages/86/91/dedad8403e7b0036d99be4878987693b7b7f62097eb8537fa6ce62ea131c/tensorflow-2.21.0-cp313-cp313-win_amd64.whl" \
  351205371
dl "onnxruntime_gpu-1.28.0-cp313-cp313-win_amd64.whl" \
  "https://files.pythonhosted.org/packages/47/e8/aab01c0b41cfc2cdcb24de9dbf546c7470e4a43d854c617220e425f2c6f5/onnxruntime_gpu-1.28.0-cp313-cp313-win_amd64.whl" \
  241464688

# ---- nvidia cu126 runtime (Linux manylinux) -----------------------------
# pip fetches wheels for the platform it runs on: on Linux this pulls the
# manylinux set; on Windows it would pull redundant win_amd64 variants (the
# Windows torch wheel already bundles the CUDA runtime). Warn if not Linux.
case "$(uname -s 2>/dev/null || echo Unknown)" in
  Linux*) nvidia_platform="manylinux" ;;
  Darwin*) nvidia_platform="macos (none exist for cu12 — skipping is correct)" ;;
  *) nvidia_platform="win (redundant — Windows torch bundles CUDA in-wheel)" ;;
esac
echo "=== nvidia cu126 runtime — detected platform: $nvidia_platform ==="
if command -v python >/dev/null 2>&1 && python -c "import sys" 2>/dev/null; then
  python -m pip download -r nvidia-cu126-linux.txt -d . \
    --only-binary=:all: --no-deps --disable-pip-version-check
else
  echo "SKIP nvidia cu126 runtime: python not available (run fetch-nvidia-linux.sh on a host with python)"
fi

echo
echo "=== archives inventory ==="
ls -1 *.whl 2>/dev/null | sed 's/^/  /'
echo
echo "done. Re-run anytime to resume missing files."
