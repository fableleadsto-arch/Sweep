#!/usr/bin/env bash
# Fetch the exact nvidia CUDA 12.6 Linux runtime wheels pinned by
# torch 2.9.1+cu126 (see nvidia-cu126-linux.txt for the pin list).
#
# Use on a Linux host to populate companion/vendor/archives/ for fully
# offline GPU installs (pip install torch-cu126 wheel + these wheels with
# --no-index). Safe to re-run: existing files are skipped.
#
# NOTE: run this on a Linux (x86_64) host — pip fetches wheels for the
# platform it runs on. On Windows it would download win_amd64 variants,
# which are redundant (the Windows torch wheel bundles the CUDA runtime
# in-wheel and needs none of these).
set -euo pipefail
cd "$(dirname "$0")"
python -m pip download \
  -r nvidia-cu126-linux.txt \
  -d . \
  --only-binary=:all: \
  --no-deps \
  --disable-pip-version-check
echo "nvidia cu126 wheels present in $(pwd):"
ls -1 nvidia_*.whl 2>/dev/null || true
