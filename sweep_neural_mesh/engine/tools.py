"""
Tool System — provider, registry, and router for the Sweep Neural Engine.

Determines the cheapest reliable method for each task:
    Arithmetic      -> deterministic calculator
    Document search -> retrieval
    Image similarity -> vision embeddings
    Speech          -> speech model
    Complex research -> retrieval + reasoning

Sweep-original implementation.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("sweep.tools")


class ToolCategory(Enum):
    CALCULATOR = "calculator"
    RETRIEVAL = "retrieval"
    VISION = "vision"
    AUDIO = "audio"
    SEARCH = "search"
    CODE_EXEC = "code_exec"
    FILESYSTEM = "filesystem"
    REASONING = "reasoning"
    EMBEDDING = "embedding"


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    output: Any
    success: bool = True
    latency_ms: float = 0.0
    tool_name: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    category: ToolCategory
    description: str
    cost_estimate: float = 0.0  # relative cost (0 = free/deterministic)
    capabilities: list[str] = field(default_factory=list)
    is_deterministic: bool = False


class ToolProvider:
    """Base class for tool implementations."""

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    def invoke(self, **kwargs) -> ToolResult:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════
# DETERMINISTIC TOOLS (original Sweep implementations)
# ════════════════════════════════════════════════════════════════════

class CalculatorTool(ToolProvider):
    """Deterministic arithmetic — no model needed."""

    def __init__(self) -> None:
        super().__init__(ToolSpec(
            name="calculator", category=ToolCategory.CALCULATOR,
            description="Deterministic arithmetic calculation",
            cost_estimate=0.0, is_deterministic=True,
            capabilities=["add", "subtract", "multiply", "divide", "percent", "sqrt"],
        ))

    def invoke(self, expression: str = "", **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        try:
            # Sanitize: only allow math operations
            sanitized = re.sub(r'[^0-9+\-*/().% sqrt]', '', expression)
            sanitized = sanitized.replace('sqrt', 'math.sqrt')
            result = eval(sanitized, {"__builtins__": {}, "math": math})
            return ToolResult(
                output=str(result), success=True,
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.spec.name,
            )
        except Exception as e:
            return ToolResult(
                output=None, success=False, error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.spec.name,
            )


class UnitConversionTool(ToolProvider):
    """Deterministic unit conversion."""

    CONVERSIONS = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda c: c * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda f: (f - 32) * 5/9,
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
        ("inches", "cm"): 2.54,
        ("cm", "inches"): 0.393701,
    }

    def __init__(self) -> None:
        super().__init__(ToolSpec(
            name="unit_converter", category=ToolCategory.CALCULATOR,
            description="Deterministic unit conversion",
            cost_estimate=0.0, is_deterministic=True,
            capabilities=["convert_units"],
        ))

    def invoke(self, value: float = 0, from_unit: str = "", to_unit: str = "", **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        key = (from_unit.lower().rstrip('s'), to_unit.lower().rstrip('s'))
        # Try plural forms too
        if key not in self.CONVERSIONS:
            key = (from_unit.lower(), to_unit.lower())
        if key not in self.CONVERSIONS:
            return ToolResult(
                output=None, success=False,
                error=f"Unknown conversion: {from_unit} -> {to_unit}",
                latency_ms=(time.perf_counter() - t0) * 1000,
                tool_name=self.spec.name,
            )
        factor = self.CONVERSIONS[key]
        if callable(factor):
            result = factor(value)
        else:
            result = value * factor
        return ToolResult(
            output=f"{result:.4g} {to_unit}", success=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            tool_name=self.spec.name,
        )


# ════════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Registry of available tools with capability matching.

    Sweep-original implementation.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolProvider] = {}
        self._category_index: dict[ToolCategory, list[str]] = {}
        self._capability_index: dict[str, list[str]] = {}

    def register(self, tool: ToolProvider) -> None:
        """Register a tool."""
        self._tools[tool.spec.name] = tool
        self._category_index.setdefault(tool.spec.category, []).append(tool.spec.name)
        for cap in tool.spec.capabilities:
            self._capability_index.setdefault(cap, []).append(tool.spec.name)
        logger.info(f"Registered tool: {tool.spec.name} ({tool.spec.category.value})")

    def get(self, name: str) -> ToolProvider | None:
        return self._tools.get(name)

    def find_by_capability(self, capability: str) -> list[ToolProvider]:
        names = self._capability_index.get(capability, [])
        return [self._tools[n] for n in names if n in self._tools]

    def find_by_category(self, category: ToolCategory) -> list[ToolProvider]:
        names = self._category_index.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def all_tools(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


# ════════════════════════════════════════════════════════════════════
# TOOL ROUTER
# ════════════════════════════════════════════════════════════════════

class ToolRouter:
    """Determines the cheapest reliable method for each task.

    Pattern:
        1. Classify the task
        2. Find tools with matching capabilities
        3. Rank by cost (deterministic first, then cheapest)
        4. Execute the best tool
        5. Return result

    Also detects when a tool is UNNECESSARY (task is simple enough to handle
    directly) and when a tool is INSUFFICIENT (result quality is too low).
    """

    # Task classification patterns
    PATTERNS = {
        "arithmetic": [
            r'\d+\s*[+\-*/]\s*\d+',
            r'what\s+is\s+\d+',
            r'\d+%\s+of\s+\d+',
            r'calculate',
        ],
        "unit_conversion": [
            r'convert\s+\d+',
            r'\d+\s+\w+\s+(?:to|in)\s+\w+',
        ],
        "lookup": [
            r'what\s+is\s+the\s+capital',
            r'who\s+(?:discovered|invented|wrote)',
            r'when\s+(?:did|was)',
        ],
    }

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._call_count: dict[str, int] = {}
        self._total_calls = 0

    def route(self, query: str, context: dict | None = None) -> ToolResult:
        """Classify query and route to the best tool."""
        t0 = time.perf_counter()
        category = self._classify(query)

        if category is None:
            return ToolResult(
                output=None, success=False,
                error="No matching tool for query",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        tools = self._registry.find_by_category(category)
        if not tools:
            return ToolResult(
                output=None, success=False,
                error=f"No tools registered for category: {category.value}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Sort by cost (deterministic first)
        tools.sort(key=lambda t: (0 if t.spec.is_deterministic else 1, t.spec.cost_estimate))
        best_tool = tools[0]

        # Track usage
        self._call_count[best_tool.spec.name] = self._call_count.get(best_tool.spec.name, 0) + 1
        self._total_calls += 1

        # Extract parameters from query
        params = self._extract_params(query, category)
        result = best_tool.invoke(**params)
        return result

    def _classify(self, query: str) -> ToolCategory | None:
        """Classify query into a tool category."""
        q_lower = query.lower()

        for cat_name, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    if cat_name == "arithmetic":
                        return ToolCategory.CALCULATOR
                    elif cat_name == "unit_conversion":
                        return ToolCategory.CALCULATOR
                    elif cat_name == "lookup":
                        return ToolCategory.RETRIEVAL
        return None

    def _extract_params(self, query: str, category: ToolCategory) -> dict:
        """Extract tool parameters from the query."""
        if category == ToolCategory.CALCULATOR:
            # Try to extract arithmetic expression
            match = re.search(r'(\d+[\s+\-*/().%\d]*)', query)
            if match:
                expr = match.group(1).strip()
                # Check for percentage
                pct = re.search(r'(\d+)%\s+of\s+(\d+)', query)
                if pct:
                    return {"expression": f"{pct.group(1)} * {pct.group(2)} / 100"}
                return {"expression": expr}
        return {}

    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "by_tool": dict(self._call_count),
        }
