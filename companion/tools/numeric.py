"""Numerical / mathematical capability — NumPy + SciPy.

Covers descriptive statistics, linear algebra, transforms and lightweight
simulations. Everything is imported lazily so a simple chat never touches the
scientific stack.
"""

from __future__ import annotations

import re
from typing import Any

from .common import as_matrix, as_numbers, load


def run_numeric(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a numerical/mathematical operation over the provided data."""
    data = payload.get("data")
    params = payload.get("params") or {}
    operation = str(params.get("operation") or _detect_operation(payload.get("task") or "")).lower()

    np = load("numpy")

    matrix = as_matrix(data)
    if operation in {"solve", "inverse", "det", "eigen"} and matrix is not None:
        return _linear_algebra(np, operation, matrix, params)

    values = as_numbers(data)
    if not values:
        raise ValueError(
            "No numeric data found. Pass a list of numbers (e.g. data=[1,2,3]) "
            "or `operation: 'solve'` with a square matrix."
        )

    np_arr = np.array(values)

    if operation in {"sum", "mean", "median", "std", "variance", "min", "max"}:
        return {
            "result": {operation: float(getattr(np_arr, operation)())},
            "summary": f"{operation.capitalize()} of {len(values)} values: {getattr(np_arr, operation)():g}",
            "libraries_used": ["numpy"],
        }

    if operation == "fft":
        fft = np.abs(np.fft.rfft(np_arr)).tolist()
        return {
            "result": {"fft_magnitudes": fft},
            "summary": f"FFT magnitude spectrum over {len(values)} samples ({len(fft)} bins).",
            "libraries_used": ["numpy"],
        }

    # Default: full descriptive profile.
    scipy = load("scipy")
    desc = {
        "count": int(len(values)),
        "sum": float(np_arr.sum()),
        "mean": float(np_arr.mean()),
        "median": float(np.median(np_arr)),
        "std": float(np_arr.std()),
        "variance": float(np_arr.var()),
        "min": float(np_arr.min()),
        "max": float(np_arr.max()),
        "range": float(np_arr.max() - np_arr.min()),
        "q1": float(np.percentile(np_arr, 25)),
        "q3": float(np.percentile(np_arr, 75)),
    }
    if len(values) >= 3:
        desc["skewness"] = float(scipy.stats.skew(np_arr))
        desc["kurtosis"] = float(scipy.stats.kurtosis(np_arr))
    summary = (
        f"{len(values)} values — mean {desc['mean']:.4g}, median {desc['median']:.4g}, "
        f"std {desc['std']:.4g}, min {desc['min']:.4g}, max {desc['max']:.4g}."
    )
    return {
        "result": {"stats": desc},
        "summary": summary,
        "libraries_used": ["numpy", "scipy"],
    }


def run_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a canned numerical simulation (random walk, Monte Carlo, growth)."""
    params = payload.get("params") or {}
    kind = str(params.get("simulation") or _detect_simulation(payload.get("task") or "")).lower()
    seed = int(params.get("seed") or 0)
    steps = int(params.get("steps") or 500)

    np = load("numpy")
    rng = np.random.default_rng(seed)

    if kind in {"random-walk", "walk"}:
        drift = float(params.get("drift") or 0.0)
        walk = rng.normal(drift, 1.0, steps)
        path = list(np.cumsum(walk))
        return {
            "result": {"kind": "random-walk", "series": path, "final_position": path[-1], "drift": drift},
            "summary": (
                f"Random walk over {steps} steps with drift {drift:g} — "
                f"final position {path[-1]:.3g}."
            ),
            "libraries_used": ["numpy"],
        }

    if kind in {"monte-carlo-pi", "monte-carlo"}:
        samples = int(params.get("samples") or steps)
        inside = 0
        for _ in range(samples):
            x, y = rng.random(2)
            if x * x + y * y <= 1.0:
                inside += 1
        estimate = 4.0 * inside / samples
        return {
            "result": {"kind": "monte-carlo-pi", "samples": samples, "estimate": estimate},
            "summary": (
                f"Monte Carlo π estimate from {samples} samples: {estimate:.4f} "
                f"(true π ≈ {3.141592653589793:.4f})."
            ),
            "libraries_used": ["numpy"],
        }

    if kind in {"logistic-growth", "logistic"}:
        pop0 = float(params.get("population") or 10)
        growth = float(params.get("growth") or 0.2)
        capacity = float(params.get("carrying_capacity") or 1000)
        pop = pop0
        series = [pop0]
        for _ in range(steps):
            pop = pop + growth * pop * (1 - pop / capacity)
            series.append(float(pop))
        return {
            "result": {"kind": "logistic-growth", "series": series, "final_population": series[-1]},
            "summary": (
                f"Logistic growth from {pop0:g} toward capacity {capacity:g} "
                f"after {steps} steps: {series[-1]:.3g}."
            ),
            "libraries_used": ["numpy"],
        }

    if kind in {"oscillator", "superposition"}:
        freqs = params.get("frequencies") or [1.0, 2.5, 5.0]
        amps = params.get("amplitudes") or [1.0, 0.5, 0.25]
        t = np.linspace(0, 2 * np.pi * float(params.get("periods") or 2), steps)
        series = np.zeros(steps)
        for f, a in zip(freqs, amps):
            series = series + float(a) * np.sin(float(f) * t)
        return {
            "result": {"kind": "oscillator", "series": list(series)},
            "summary": f"Superposition of {len(freqs)} sine waves over {steps} samples.",
            "libraries_used": ["numpy"],
        }

    raise ValueError(
        "Unknown simulation. Use one of: random-walk, monte-carlo-pi, logistic-growth, oscillator."
    )


def _linear_algebra(np, operation: str, matrix: list[list[float]], params: dict[str, Any]) -> dict[str, Any]:
    m = np.array(matrix)
    summary = f"Matrix {m.shape[0]}×{m.shape[1]}."
    if operation == "solve":
        b = params.get("rhs") or params.get("b")
        if b is None or not isinstance(b, (list, tuple)):
            raise ValueError("`solve` needs a right-hand side via params.rhs (list of numbers).")
        rhs = np.array([float(v) for v in b])
        solution = np.linalg.solve(m, rhs)
        return {
            "result": {"solution": [float(v) for v in solution]},
            "summary": f"{summary} Linear system solution: {', '.join(f'{v:.4g}' for v in solution)}.",
            "libraries_used": ["numpy"],
        }
    if operation == "inverse":
        return {
            "result": {"inverse": [[float(v) for v in row] for row in np.linalg.inv(m)]},
            "summary": f"{summary} Matrix inverse computed.",
            "libraries_used": ["numpy"],
        }
    if operation == "det":
        return {
            "result": {"determinant": float(np.linalg.det(m))},
            "summary": f"{summary} Determinant = {float(np.linalg.det(m)):.4g}.",
            "libraries_used": ["numpy"],
        }
    if operation == "eigen":
        values, vectors = np.linalg.eig(m)
        return {
            "result": {
                "eigenvalues": [complex(v) for v in values],
                "eigenvectors": [[complex(c) for c in row] for row in vectors],
            },
            "summary": f"{summary} Eigenvalues: {', '.join(f'{v.real:.3g}{'+'+v.imag:.2g}i' if v.imag else f'{v.real:.3g}' for v in values)}.",
            "libraries_used": ["numpy"],
        }
    raise ValueError(f"Unknown matrix operation: {operation}")


def _detect_operation(task: str) -> str:
    text = task.lower()
    if re.search(r"solve|equation|system", text):
        return "solve"
    if "fft" in text or "fourier" in text:
        return "fft"
    if "inverse" in text:
        return "inverse"
    if "determinant" in text or "det " in text:
        return "det"
    if "eigen" in text:
        return "eigen"
    return "describe"


def _detect_simulation(task: str) -> str:
    text = task.lower()
    if "monte" in text or "pi" == text.strip():
        return "monte-carlo-pi"
    if "walk" in text or "drift" in text:
        return "random-walk"
    if "logistic" in text or "growth" in text:
        return "logistic-growth"
    if "oscillat" in text or "wave" in text:
        return "oscillator"
    return "random-walk"
