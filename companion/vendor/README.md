# Vendored AI/ML frameworks

Relay ships the source of the largest AI/ML frameworks **inside this repo** so
the brain service never has to re-fetch them and can work from local source.
Three tiers:

1. **Importable packages** (`transformers/`, `langchain/`, ...) — full pure-Python
   source trees extracted from the official PyPI source distributions. When the
   framework is *not* installed in the environment, the capability engine
   imports it from here (installed versions always win).
2. **Extracted source trees** (`torch/`, `tensorflow/`) — complete upstream
   repository sources unpacked next to their tarball for browsing/grepping and
   offline builds. These are **not importable** (both need a compiled native
   backend); use the pip wheel / bundled wheel instead.
3. **Source archives** (`archives/*.tar.gz`) — full source tarballs of the
   compiled / CUDA / proprietary giants (PyTorch, TensorFlow, vLLM, JAX,
   Keras). Their Python wheels require a native build, so the source is stored
   for offline builds instead of vendored as importable trees.

## Extracted source trees (non-importable)

| Package    | Version | Path           | Lines      | License      | Extracted from                    |
|------------|---------|----------------|------------|--------------|-----------------------------------|
| PyTorch    | 2.9.1   | `torch/`       | 3,780,234  | BSD-3-Clause | `archives/torch-v2.9.1-src.tar.gz` |
| TensorFlow | 2.21.0  | `tensorflow/`  | (large)    | Apache-2.0   | `archives/tensorflow-v2.21.0-src.tar.gz` |

Notes on the PyTorch tree (`torch/`, tag v2.9.1, 19,391 files):

- Layout: `torch/` (Python API + generated code), `aten/` + `c10/` (C++/CUDA
  tensor core), `torchgen/` (code generator), `functorch/`, `caffe2/`,
  `android/`, `test/`, plus build system (`cmake/`, `setup.py`, Bazel/Buck).
- The GitHub tarball stores 4 entries as symlinks; on Windows they were
  materialized as real copies: `.dockerignore` ← `.gitignore`,
  `.github/ci_commit_pins/triton.txt`, `docs/requirements.txt`, and
  `functorch/docs/source/notebooks/`.
- Building from this tree requires a full native toolchain (CUDA toolkit,
  MSVC/clang, etc.) — for actual use install the bundled cu126 wheel above.


## Importable packages (fallback import path)

| Package       | Version   | License      | Source                               |
|---------------|-----------|--------------|--------------------------------------|
| transformers  | 5.15.0    | Apache-2.0   | https://pypi.org/project/transformers |
| langchain     | 1.3.14    | MIT          | https://pypi.org/project/langchain    |
| llama_index   | 0.14.23   | MIT          | https://pypi.org/project/llama-index-core |
| autogen       | 0.14.1    | Apache-2.0   | https://pypi.org/project/autogen      |
| crewai        | 1.15.14   | MIT          | https://pypi.org/project/crewai       |

Each directory keeps its upstream `LICENSE` file. `llama_index` is a
namespace package (no top-level `__init__.py`) — that is its native layout.

## Source archives (offline build sources)

| Package     | Version  | License        | Archive file                  | Source                                        |
|-------------|----------|----------------|-------------------------------|-----------------------------------------------|
| PyTorch     | 2.9.1    | BSD-3-Clause   | `torch-v2.9.1-src.tar.gz`     | https://github.com/pytorch/pytorch (tag v2.9.1) |
| TensorFlow  | 2.21.0   | Apache-2.0     | `tensorflow-v2.21.0-src.tar.gz` | https://github.com/tensorflow/tensorflow (tag v2.21.0) |
| vLLM        | 0.27.0   | Apache-2.0     | `vllm-0.27.0.tar.gz`          | https://pypi.org/project/vllm                  |
| JAX         | 0.11.0   | Apache-2.0     | `jax-0.11.0.tar.gz`           | https://pypi.org/project/jax                   |
| Keras       | 3.15.1   | Apache-2.0     | `keras-3.15.1.tar.gz`         | https://pypi.org/project/keras                 |

PyTorch and TensorFlow publish **no sdist on PyPI** (wheels only), so their
source comes from the GitHub release tags.

## Bundled wheels (local-only, not committed)

Pre-built platform wheels stored in `archives/` so the exact binary bundle is
available locally without re-downloading. These are **multi-GB files** —
GitHub rejects pushes over 100MB per file, so they are listed in `.gitignore`
and only exist in the working tree. `vendor_loader.py` reports them as
present/missing so Relay always knows what is actually on disk.

| Bundle                | Version         | Files (Windows / cp313)                                       | Size   |
|-----------------------|-----------------|--------------------------------------------------------------|--------|
| PyTorch CUDA (torch)  | 2.9.1+cu126     | `torch-2.9.1+cu126-cp313-cp313-win_amd64.whl`                 | 2.46GB |
| torchvision           | 0.24.1+cu126    | `torchvision-0.24.1+cu126-cp313-cp313-win_amd64.whl`          | 8.4MB  |
| torchaudio            | 2.9.1+cu126     | `torchaudio-2.9.1+cu126-cp313-cp313-win_amd64.whl`            | 2.0MB  |
| TensorFlow            | 2.21.0          | `tensorflow-2.21.0-cp313-cp313-win_amd64.whl` (CPU — Windows  | 351MB  |
|                       |                 |  ships no GPU build; use Linux/WSL for `tensorflow[and-cuda]`) |        |
| ONNX Runtime GPU      | 1.28.0          | `onnxruntime_gpu-1.28.0-cp313-cp313-win_amd64.whl` (CUDA EP)   | 241MB  |

Re-download on another machine (or `pip install` directly):

```bash
pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
pip install tensorflow==2.21.0 onnxruntime-gpu==1.28.0
```

The PyTorch wheels come from the official PyTorch CUDA 12.6 index; TensorFlow
and ONNX Runtime from PyPI. vLLM and bitsandbytes ship **no Windows wheels**
(Linux-only) — their source archives above cover offline Linux builds.

### nvidia CUDA 12.6 runtime (Linux-only, for offline GPU installs)

The Windows torch wheel bundles the CUDA runtime in-wheel and needs **none**
of these. On **Linux** (e.g. `Dockerfile.brain` on an NVIDIA host), torch
2.9.1+cu126 declares 15 nvidia runtime wheels as exact pins — stored here so
`pip install --no-index` works fully offline:

- `nvidia-cuda-runtime-cu12==12.6.77`, `nvidia-cuda-nvrtc-cu12==12.6.77`,
  `nvidia-cuda-cupti-cu12==12.6.80`, `nvidia-nvjitlink-cu12==12.6.85`,
  `nvidia-nvtx-cu12==12.6.77`, `nvidia-cublas-cu12==12.6.4.1`,
  `nvidia-cudnn-cu12==9.10.2.21`, `nvidia-cufft-cu12==11.3.0.4`,
  `nvidia-curand-cu12==10.3.7.77`, `nvidia-cusolver-cu12==11.7.1.2`,
  `nvidia-cusparse-cu12==12.5.4.2`, `nvidia-cusparselt-cu12==0.7.1`,
  `nvidia-nccl-cu12==2.27.5`, `nvidia-nvshmem-cu12==3.3.20`,
  `nvidia-cufile-cu12==1.11.1.6`

The pin list lives in `archives/nvidia-cu126-linux.txt`; re-fetch all of them
on any host with `./archives/fetch-nvidia-linux.sh`. All wheels are gitignored
(they are Linux-only binaries and exceed GitHub's per-file size limits).

### Universal bundle sync

`archives/sync-bundles.sh` fetches **every** stored wheel (Windows + nvidia
Linux) into the archive dir on whatever host it runs on — safe to re-run
(existing files are kept, partial ones resume with `curl -C -`). `--force`
re-downloads everything. This is the one-command way to repopulate a fresh
checkout or a deployment host.

## Loader

`companion/tools/vendor_loader.py` exposes:

- `vendored_packages()` — names of importable vendored packages
- `vendored_path(name)` — path of an importable vendored package (or `None`)
- `source_archives()` — metadata for every stored archive
- `archive_path(name)` — path of an archive tarball
- `bundled_wheels()` — metadata for every known bundled wheel
- `wheel_path(name)` — path of a stored bundled wheel (or `None`)
- `add_vendored_paths()` — idempotently adds importable vendored packages to
  `sys.path` (append-only, so pip-installed versions keep priority)

`companion/tools/common.py::load()` and `module_available()` consult the
loader: a framework is considered *available* when it is pip-installed **or**
vendored, and importing falls back to the vendored source when needed.

## Updating

Re-run the fetch script (see `.github`/scripts if present) or manually:

```bash
mkdir -p companion/vendor/archives
pip download --no-deps --no-binary :all: -d /tmp/vend <pkg>
curl -L -o companion/vendor/archives/torch-<ver>-src.tar.gz \
  https://codeload.github.com/pytorch/pytorch/tar.gz/refs/tags/v<ver>
```

Keep the tables above in sync with whatever is actually stored.
