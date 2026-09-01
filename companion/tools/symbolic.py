"""Symbolic mathematics capability — SymPy.

Parses an expression/equation string and performs solve / simplify /
differentiate / integrate / factor / expand / limit. Lazy import.
"""

from __future__ import annotations

import re
from typing import Any

from .common import as_text, load


def run_symbolic(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve or manipulate a symbolic expression."""
    params = payload.get("params") or {}
    expression = str(params.get("expression") or params.get("equation") or "").strip()
    if not expression:
        expression = as_text(payload.get("data"), params).strip()
    if not expression:
        raise ValueError(
            "No expression found. Pass params.expression (e.g. 'x**2 - 4 = 0') "
            "or `data` as the expression string."
        )

    sympy = load("sympy")

    operation = str(params.get("operation") or _detect_operation(payload.get("task") or "")).lower()
    is_equation = "=" in expression

    if is_equation and operation in {"differentiate", "integrate"}:
        # An equation is not differentiable directly; solve first and tell the user.
        raise ValueError(
            f"Cannot {operation} an equation directly — provide a bare expression "
            f"like 'x**2 + 3*x' for {operation}."
        )

    if is_equation:
        lhs, rhs = expression.split("=", 1)
        expr = sympy.sympify(lhs.strip()) - sympy.sympify(rhs.strip())
        symbol = _primary_symbol(expr, params, sympy)
        solved = sympy.solve(expr, symbol)
        solutions = [_sympy_text(s, sympy) for s in solved]
        result = {"type": "equation", "equation": expression, "solutions": solutions}
        summary = (
            f"Solved {expression} — solutions: "
            f"{', '.join(solutions) if solutions else 'no real solutions'}."
        )
        return {"result": result, "summary": summary, "libraries_used": ["sympy"]}

    expr = sympy.sympify(expression)

    if operation == "simplify":
        result = _sympy_text(sympy.simplify(expr), sympy)
    elif operation == "factor":
        result = _sympy_text(sympy.factor(expr), sympy)
    elif operation == "expand":
        result = _sympy_text(sympy.expand(expr), sympy)
    elif operation == "differentiate":
        symbol = _primary_symbol(expr, params, sympy)
        order = int(params.get("order") or 1)
        derivative = expr
        for _ in range(order):
            derivative = sympy.diff(derivative, symbol)
        result = _sympy_text(derivative, sympy)
    elif operation == "integrate":
        symbol = _primary_symbol(expr, params, sympy)
        antiderivative = sympy.integrate(expr, symbol)
        result = _sympy_text(antiderivative, sympy)
    elif operation == "limit":
        symbol = _primary_symbol(expr, params, sympy)
        at = sympy.sympify(str(params.get("at") or "oo"))
        limit = sympy.limit(expr, symbol, at)
        result = _sympy_text(limit, sympy)
    elif operation == "evaluate":
        result = _sympy_text(expr, sympy)
    else:
        # Default: try to solve for a symbol, else simplify.
        symbol = _primary_symbol(expr, params, sympy)
        solved = sympy.solve(expr, symbol)
        if solved:
            result = {"solutions": [_sympy_text(s, sympy) for s in solved]}
        else:
            result = _sympy_text(sympy.simplify(expr), sympy)

    return {
        "result": {"type": operation, "expression": expression, "result": result},
        "summary": f"SymPy {operation}: {result}",
        "libraries_used": ["sympy"],
    }


def _primary_symbol(expr, params: dict[str, Any], sympy) -> Any:
    """Pick the symbol to operate on (explicit param, else the first free var)."""
    explicit = str(params.get("symbol") or "").strip()
    if explicit:
        return sympy.Symbol(explicit)
    free = sorted(expr.free_symbols, key=lambda s: s.name)
    if free:
        return free[0]
    raise ValueError("No symbols found in the expression.")


def _sympy_text(value, sympy) -> str:
    """Render a SymPy result as plain text (with nice math where sensible)."""
    if isinstance(value, str):
        return value
    try:
        import sympy.printing.pretty  # noqa: F401

        return sympy.sstr(value, sympy_numeric=True, full_prec=False)
    except Exception:  # noqa: BLE001 - formatting is best-effort
        return str(value)


def _detect_operation(task: str) -> str:
    text = task.lower()
    if re.search(r"differentiat|derivative|derive", text):
        return "differentiate"
    if re.search(r"integrat|antiderivative", text):
        return "integrate"
    if "simplify" in text:
        return "simplify"
    if "factor" in text:
        return "factor"
    if "expand" in text:
        return "expand"
    if "limit" in text:
        return "limit"
    return "solve"
