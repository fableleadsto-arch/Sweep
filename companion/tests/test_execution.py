"""Tests for the sandboxed Python executor (spec §35)."""

from __future__ import annotations

import asyncio

import pytest

from companion.execution import (
    execute_python,
    run_python_code,
    scan_python_code,
)

SIMPLE = """\
import json
result = {"total": sum(env.get("values", [])), "n": len(env.get("values", []))}
"""


def test_scan_allows_scientific_stack() -> None:
    violations = scan_python_code(
        "import numpy as np\n"
        "from sympy import symbols\n"
        "from pandas import DataFrame\n"
        "from sklearn.cluster import KMeans\n"
        "import matplotlib.pyplot as plt\n"
        "import networkx as nx\n"
        "from faker import Faker\n"
        "import math, statistics, random, datetime, json, re\n"
    )
    assert violations == []


def test_scan_allows_compute_only_deep_learning_frameworks() -> None:
    # The sandbox permits compute-only framework imports (torch, TF, JAX,
    # onnxruntime) while blocking every I/O surface.
    violations = scan_python_code(
        "import torch\n"
        "import tensorflow as tf\n"
        "import onnxruntime as ort\n"
        "result = {'ok': True}\n"
    )
    assert violations == []


def test_scan_refuses_model_downloading_frameworks() -> None:
    # transformers / diffusers / timm / accelerate hit the Hugging Face hub
    # (network egress) so they stay OUT of the sandbox — those go through the
    # capability engine instead, which has explicit model-download policy.
    for code in (
        "from transformers import AutoModel\nresult = {'ok': True}\n",
        "from diffusers import StableDiffusionPipeline\nresult = {'ok': True}\n",
        "import timm\nresult = {'ok': True}\n",
    ):
        assert scan_python_code(code), f"expected {code!r} to be rejected"


@pytest.mark.parametrize(
    "code",
    [
        "import os\n",
        "import subprocess\n",
        "import socket\n",
        "import ctypes\n",
        "import importlib\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "import requests\n",
        "import sys\n",
        # I/O-heavy frameworks stay OUT of the sandbox allowlist even though
        # they are capabilities — the sandbox is compute-only.
        "import vllm\n",
        "import deepspeed\n",
        "import llama_cpp\n",
        "import qdrant_client\n",
        "import pymilvus\n",
        "import crewai\n",
    ],
)
def test_scan_refuses_system_imports(code: str) -> None:
    assert scan_python_code(code), f"expected {code!r} to be rejected"


@pytest.mark.parametrize(
    "code",
    [
        "__import__('os').system('id')",
        "eval('1+1')",
        "exec('x = 1')",
        "open('/etc/passwd').read()",
        "().__class__.__mro__",
        "getattr(__builtins__, 'exec')",
    ],
)
def test_scan_refuses_escape_patterns(code: str) -> None:
    assert scan_python_code(code), f"expected {code!r} to be rejected"


def test_basic_execution_with_env() -> None:
    out = asyncio.run(execute_python(SIMPLE, env={"values": [1, 2, 3]}, timeout_ms=10_000))
    assert out["ok"] is True
    assert out["result"] == {"total": 6, "n": 3}
    assert out["error"] == ""
    assert out["sandboxed"] is True
    assert out["duration_ms"] >= 0


def test_uses_real_library_underneath() -> None:
    code = (
        "import numpy as np\n"
        "import json\n"
        "a = np.array(env['a'])\n"
        "b = np.array(env['b'])\n"
        "result = {\"dot\": float(np.dot(a, b)), \"shape\": list(np.add(a, b))}\n"
    )
    out = run_python_code(code, env={"a": [1, 2, 3], "b": [4, 5, 6]}, timeout_ms=15_000)
    assert out["ok"] is True
    assert out["result"] == {"dot": 32.0, "shape": [5, 7, 9]}


def test_symbolic_math_via_sympy() -> None:
    code = (
        "from sympy import symbols, solve, Eq\n"
        "import json\n"
        "x = symbols('x')\n"
        "result = [str(s) for s in solve(Eq(x**2 - 4, 0), x)]\n"
    )
    out = run_python_code(code, timeout_ms=15_000)
    assert out["ok"] is True
    assert set(out["result"]) == {"-2", "2"}


def test_rejects_disallowed_import_even_with_valid_logic() -> None:
    out = run_python_code("import os\nresult = {'x': 1}\n")
    assert out["ok"] is False
    assert out["violations"] and "import 'os'" in out["violations"][0]
    assert "Sandbox policy" in out["error"]


def test_times_out_on_infinite_loop() -> None:
    code = "while True:\n    pass\n"
    out = run_python_code(code, timeout_ms=2_000)
    assert out["ok"] is False
    assert "timed out" in out["error"]


def test_script_runtime_error_is_reported_not_raised() -> None:
    out = run_python_code("x = 1 / 0\nresult = {'x': x}\n")
    assert out["ok"] is False
    assert "exited with code" in out["error"] or "ZeroDivisionError" in out["error"]


def test_stdout_is_captured() -> None:
    out = run_python_code("print('hello from sandbox')\nresult = {'done': True}\n")
    assert out["ok"] is True
    assert "hello from sandbox" in out["stdout"]


def test_syntax_error_reported_as_violation() -> None:
    assert scan_python_code("def broken(:\n")
