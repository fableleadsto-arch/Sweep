"""Sandboxed Python execution for Relay (spec §35).

Runs *generated* Python — the scientific-stack code an agent writes to solve a
problem with NumPy/SymPy/Pandas/scikit-learn/etc — inside a disposable
subprocess with hard limits:

  - AST import allowlist: only data/scientific stdlib + the scientific stack
    can be imported. Anything else (os, sys, socket, ctypes, importlib,
    subprocess, ...) is rejected before the process even starts.
  - Text-level escape scan (defense in depth): `__import__`, `eval(`, `exec(`,
    `open(`, `.__class__`, `.__globals__`, `getattr` etc. are refused.
  - POSIX resource limits via ``resource.setrlimit``: CPU seconds, address
    space, file size, process count. On platforms without ``resource``
    (Windows dev), the subprocess timeout is the enforcement backstop.
  - The process runs with `-I -B -E -u` (isolated, no site-packages dir
    manipulation beyond the interpreter's own, no bytecode writes), in an
    empty temp working directory, with a scrubbed environment (no secrets).
  - stdout/stderr are captured and capped; the result is a single JSON line
    (`__RELAY_RESULT__ <json>`) a caller can parse.

This is a practical sandbox for trusted-relay-generated code, not a security
boundary for arbitrary hostile input: it raises the cost of escape and caps
blast radius, but a determined adversary with code execution could still find
gaps. Deploy it behind the same auth gate as every other brain route.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

# ── policy ──────────────────────────────────────────────────────────────────

MAX_CODE_CHARS = 50_000
DEFAULT_TIMEOUT_MS = 30_000
MAX_TIMEOUT_MS = 120_000
OUTPUT_CAP = 200_000

# Top-level modules an agent's generated script may import. Everything else is
# refused before execution. Allowlist beats denylist: new capabilities are an
# explicit, reviewed addition instead of an arms race.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        # data / science
        "numpy",
        "scipy",
        "pandas",
        "sympy",
        "sklearn",
        "matplotlib",
        "PIL",
        "networkx",
        "faker",
        # deep learning / ML frameworks (compute-only imports; the sandbox
        # still blocks network, filesystem and process access). Transformers /
        # diffusers / timm / accelerate deliberately stay OUT: they fetch model
        # weights from the Hugging Face hub (network egress) and belong to the
        # capability engine instead, which has explicit model-download policy.
        "torch",
        "tensorflow",
        "keras",
        "jax",
        "onnxruntime",
        "sklearn",
        # pure computation stdlib
        "json",
        "math",
        "random",
        "statistics",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "re",
        "string",
        "uuid",
        "decimal",
        "fractions",
        "operator",
        "heapq",
        "bisect",
        "array",
        "enum",
        "textwrap",
        "difflib",
        "unicodedata",
        "hashlib",
        "base64",
        "binascii",
        "typing",
        "numbers",
        "dataclasses",
        "copy",
        "warnings",
    }
)

# Escape-hatch substrings refused in the source text. These catch dynamic
# loading and object-graph walking that the import allowlist alone can't.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "__import__",
    "__builtins__",
    "__subclasses__",
    "__globals__",
    "__mro__",
    "__base__",
    "__bases__",
    ".__class__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "input(",
    "breakpoint()",
    "getattr(",
    "setattr(",
    "delattr(",
    "vars(",
    "globals()",
    "locals()",
)


def scan_python_code(code: str) -> list[str]:
    """Return every policy violation in the source, else an empty list.

    Two independent passes:

      1. AST import walk — collects the top-level module of every `import` and
         `from` statement and rejects any module outside ALLOWED_IMPORTS.
      2. Text scan — rejects dynamic loading / object-graph escape substrings
         (catches `__import__("os")`, `().__class__...`, ``eval``/``exec``).
    """
    violations: list[str] = []
    if len(code) > MAX_CODE_CHARS:
        violations.append(f"code too large ({len(code)} chars, max {MAX_CODE_CHARS})")

    try:
        import ast

        tree = ast.parse(code)
    except SyntaxError as exc:
        violations.append(f"syntax error: {exc.msg}")
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    violations.append(f"import '{top}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in ALLOWED_IMPORTS:
                violations.append(f"import from '{top}' is not allowed")

    lowered = code.lower()
    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in lowered:
            violations.append(f"forbidden pattern '{needle}'")

    # Dedup while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# ── the wrapper ─────────────────────────────────────────────────────────────

_HEADER = (
    "import json as __relay_json, sys as __relay_sys\n"
    "__relay_env = {}\n"
    "if len(__relay_sys.argv) > 1:\n"
    "    try:\n"
    "        with open(__relay_sys.argv[1], encoding='utf-8') as __relay_f:\n"
    "            __relay_env = __relay_json.load(__relay_f)\n"
    "    except Exception:\n"
    "        pass\n"
    "env = __relay_env\n"
)

_FOOTER = (
    "\n__relay_result = globals().get('result', None)\n"
    "if __relay_result is not None:\n"
    "    def __relay_default(__o):\n"
    "        try:\n"
    "            import numpy as __np\n"
    "        except Exception:\n"
    "            __np = None\n"
    "        if __np is not None and isinstance(__o, __np.generic):\n"
    "            return __o.item()\n"
    "        if __np is not None and isinstance(__o, __np.ndarray):\n"
    "            return __o.tolist()\n"
    "        if isinstance(__o, (bytes, bytearray)):\n"
    "            return __o.decode('utf-8', errors='replace')\n"
    "        return str(__o)\n"
    "    print('__RELAY_RESULT__ ' + __relay_json.dumps(\n"
    "        __relay_result, default=__relay_default, ensure_ascii=False)[:200000])\n"
)


def _resource_limiter(cpu_seconds: int, max_bytes: int) -> Any:
    """POSIX-only preexec fn that hard-caps CPU + RAM + file size + processes."""

    def _apply() -> None:  # pragma: no cover - platform dependent
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))

    return _apply


def _scrubbed_env() -> dict[str, str]:
    """Minimal environment for the child: no secrets, no repo paths."""
    keep = {"PATH", "HOME", "LANG", "TMPDIR", "TEMP", "TMP", "USER", "SYSTEMROOT", "WINDIR", "PYTHONIOENCODING"}
    base = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
    }
    for key in keep:
        if key in os.environ and key not in base:
            base[key] = os.environ[key]
    return base


# ── runner ──────────────────────────────────────────────────────────────────


def run_python_code(
    code: str,
    env: dict[str, Any] | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, Any]:
    """Execute one generated script in isolation and return structured output."""
    started = time.perf_counter()
    violations = scan_python_code(code)
    if violations:
        return {
            "ok": False,
            "duration_ms": 0,
            "result": None,
            "stdout": "",
            "stderr": "",
            "error": "Sandbox policy rejected the code.",
            "violations": violations[:20],
            "sandboxed": True,
        }

    timeout_ms = max(1_000, min(timeout_ms, MAX_TIMEOUT_MS))
    timeout_s = timeout_ms / 1000.0
    script = (_HEADER + "\n" + code + "\n" + _FOOTER)

    with tempfile.TemporaryDirectory(prefix="relay-exec-") as workdir:
        script_path = os.path.join(workdir, "script.py")
        env_path = os.path.join(workdir, "env.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(env or {}, f)

        preexec = _resource_limiter(30, 512 * 1024 * 1024) if os.name == "posix" else None
        proc: subprocess.Popen[bytes] | None = None
        timed_out = False
        try:
            proc = subprocess.Popen(  # noqa: S603 - deliberate isolated subprocess
                [sys.executable, "-I", "-B", "-E", "-u", script_path, env_path],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=_scrubbed_env(),
                preexec_fn=preexec,
                start_new_session=True,
            )
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout_bytes, stderr_bytes = (b"", b"")
            if proc is not None:
                _kill_process_group(proc.pid)
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001 - cleanup only
                    pass

        stdout = _decode(stdout_bytes)
        stderr = _decode(stderr_bytes)

        if timed_out:
            return {
                "ok": False,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "result": None,
                "stdout": _cap_text(stdout),
                "stderr": _cap_text(stderr),
                "error": f"Execution timed out after {timeout_s:.1f}s.",
                "violations": [],
                "sandboxed": True,
            }

        returncode = proc.returncode if proc is not None else 1
        if returncode != 0:
            return {
                "ok": False,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "result": None,
                "stdout": _cap_text(stdout),
                "stderr": _cap_text(stderr),
                "error": f"Script exited with code {returncode}. {stderr.strip()[:400]}",
                "violations": [],
                "sandboxed": True,
            }

        result, marker_error = _parse_result(stdout)
        return {
            "ok": True,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "result": result,
            "stdout": _cap_text(stdout),
            "stderr": _cap_text(stderr),
            "error": marker_error,
            "violations": [],
            "sandboxed": True,
        }


async def execute_python(
    code: str,
    env: dict[str, Any] | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, Any]:
    """Async facade so the FastAPI route never blocks the event loop."""
    return await asyncio.to_thread(run_python_code, code, env, timeout_ms)


# ── helpers ─────────────────────────────────────────────────────────────────


def _decode(raw: bytes | None) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _cap_text(text: str) -> str:
    return text[:OUTPUT_CAP]


def _parse_result(stdout: str) -> tuple[Any, str]:
    """Pull the `__RELAY_RESULT__` marker line, if any, off stdout."""
    for line in stdout.splitlines():
        if line.startswith("__RELAY_RESULT__ "):
            payload = line[len("__RELAY_RESULT__ ") :]
            try:
                return json.loads(payload), ""
            except json.JSONDecodeError as exc:
                return None, f"Result marker present but not valid JSON: {exc}"
    return None, ""


def _kill_process_group(pid: int) -> None:
    """Best-effort kill of the child's whole process group."""
    if pid is None:
        return
    try:
        os.killpg(pid, 9)  # type: ignore[attr-defined]
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        pass  # Windows / already exited
    try:
        os.kill(pid, 9)  # fallback for non-grouped children
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        pass  # Windows / already exited
