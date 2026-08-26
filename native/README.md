# Sweep Native Extension

`sweep_native` is the optional compiled fast path for Sweep's audio/visual
primitives. The Python package works without it — `sweep.fastpath`
transparently falls back to pure-Python implementations with identical
semantics.

## Build

Requires a C++17 compiler, CMake >= 3.15, and Python >= 3.11.

```bash
pip install pybind11
cmake -S native -B native/build -DPYTHON_EXECUTABLE=$(which python3)
cmake --build native/build
```

On Windows (Developer PowerShell for VS2022):

```powershell
pip install pybind11
cmake -S native -B native/build
cmake --build native/build --config Release
```

## Install

Copy (or symlink) the built module next to the `sweep` package so
`import sweep_native` resolves:

- Linux/macOS: `sweep_native.cpython-*.so`
- Windows: `Release/sweep_native.pyd`

Verify:

```bash
python -c "import sweep_native; print(sweep_native.rms_energy([3.0, 4.0]))"
python -c "from sweep.fastpath import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)"
```

## API

| Function | Signature | Notes |
| --- | --- | --- |
| `rms_energy` | `(samples: Sequence[float]) -> float` | RMS of an audio buffer |
| `moving_average` | `(values: Sequence[float], window: int) -> list[float]` | Trailing window, partial warmup |
| `frame_diff_score` | `(prev: bytes, curr: bytes) -> float` | Fraction of differing bytes in [0..1] |

The Python fallbacks live in `sweep/fastpath/fallback.py` and are the
reference semantics for these functions.
