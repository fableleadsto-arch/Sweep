"""
Failure Analysis System — §28

Whenever Sweep fails a test:
1. Save the input.
2. Save the expected result.
3. Save Sweep's result.
4. Categorize the failure.
5. Determine whether it is:
   - perception failure
   - retrieval failure
   - reasoning failure
   - tool-selection failure
   - hallucination
   - data problem
   - implementation bug
6. Add an appropriate regression test.
7. Fix the underlying problem.
8. Re-run the entire benchmark.

Never simply hard-code the expected answer.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FAILURE_CATEGORIES = [
    "KNOWLEDGE_ERROR",
    "REASONING_ERROR",
    "ARITHMETIC_ERROR",
    "RETRIEVAL_ERROR",
    "ENTITY_RESOLUTION_ERROR",
    "PERCEPTION_ERROR",
    "OCR_ERROR",
    "TOOL_SELECTION_ERROR",
    "TOOL_EXECUTION_ERROR",
    "MEMORY_ERROR",
    "HALLUCINATION",
    "SOURCE_ATTRIBUTION_ERROR",
    "INSTRUCTION_FOLLOWING_ERROR",
    "CONTEXT_LOSS",
    "OVERCONFIDENCE",
    "UNDERCONFIDENCE",
    "TIMEOUT",
    "INFRASTRUCTURE_FAILURE",
]


@dataclass
class FailureRecord:
    """A single failure record."""
    failure_id: str
    timestamp: float
    input_text: str
    expected_output: str
    actual_output: str
    category: str
    subcategory: str
    severity: str  # low, medium, high, critical
    task_domain: str
    confidence: float
    latency_ms: float
    is_regression: bool
    root_cause: str
    fix_applied: str
    regression_test_added: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailurePattern:
    """Aggregated failure pattern."""
    category: str
    count: int
    avg_confidence: float
    domains_affected: list[str]
    root_causes: list[str]
    recommendation: str


class FailureAnalyzer:
    """
    §28: Tracks, categorizes, and analyzes failures.
    Generates regression tests and prevents repetition.
    """

    def __init__(self, storage_dir: str | Path = "sweep_neural_mesh/training/failures") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._failures: list[FailureRecord] = []
        self._regression_tests: list[dict[str, Any]] = []
        self._counter = 0

    def record_failure(
        self,
        input_text: str,
        expected_output: str,
        actual_output: str,
        category: str = "",
        task_domain: str = "",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> FailureRecord:
        """Record a failure and categorize it."""
        self._counter += 1
        if not category:
            category = self._categorize_failure(input_text, expected_output, actual_output)

        severity = self._assess_severity(category, confidence)
        subcategory = self._get_subcategory(category, input_text, expected_output, actual_output)
        root_cause = self._analyze_root_cause(category, input_text, expected_output, actual_output)

        record = FailureRecord(
            failure_id=f"FAIL-{self._counter:06d}",
            timestamp=time.time(),
            input_text=input_text[:2000],
            expected_output=expected_output[:1000],
            actual_output=actual_output[:1000],
            category=category,
            subcategory=subcategory,
            severity=severity,
            task_domain=task_domain,
            confidence=confidence,
            latency_ms=latency_ms,
            is_regression=self._is_regression(category, input_text),
            root_cause=root_cause,
            fix_applied="",
            regression_test_added=False,
            metadata=metadata or {},
        )

        # Add regression test
        reg_test = self._create_regression_test(record)
        self._regression_tests.append(reg_test)
        record.regression_test_added = True

        self._failures.append(record)
        self._save_record(record)

        return record

    def _categorize_failure(
        self, input_text: str, expected: str, actual: str
    ) -> str:
        """Auto-categorize a failure based on input/output analysis."""
        input_lower = input_text.lower()
        actual_lower = actual.lower()
        expected_lower = expected.lower()

        # Arithmetic check
        if any(op in input_text for op in ["+", "-", "*", "/", "%", "calculate", "sum", "average"]):
            return "ARITHMETIC_ERROR"

        # Hallucination check: model said something that contradicts known facts
        if actual_lower not in ("unknown", "insufficient", "uncertain", "i don't know"):
            if expected_lower in ("unknown", "insufficient", "uncertain"):
                return "HALLUCINATION"

        # Reasoning error: wrong logical conclusion
        if expected_lower in ("yes", "no", "true", "false") and actual_lower not in (expected_lower, ""):
            return "REASONING_ERROR"

        # Entity resolution
        if "same entity" in input_lower or "are they" in input_lower:
            return "ENTITY_RESOLUTION_ERROR"

        # Knowledge error
        if any(kw in input_lower for kw in ["what is", "who is", "when was", "where is"]):
            return "KNOWLEDGE_ERROR"

        # Instruction following
        if len(actual) > 0 and len(expected) > 0:
            if actual_lower.strip() != expected_lower.strip():
                if len(expected) < 20:
                    return "INSTRUCTION_FOLLOWING_ERROR"

        return "REASONING_ERROR"

    def _assess_severity(self, category: str, confidence: float) -> str:
        """Assess the severity of a failure."""
        critical_categories = {"HALLUCINATION", "INFRASTRUCTURE_FAILURE"}
        high_categories = {"REASONING_ERROR", "ARITHMETIC_ERROR", "RETRIEVAL_ERROR"}

        if category in critical_categories:
            return "critical"
        if category in high_categories:
            if confidence > 0.8:
                return "critical"  # high confidence + wrong = worst
            return "high"
        if confidence > 0.7:
            return "medium"
        return "low"

    def _get_subcategory(
        self, category: str, input_text: str, expected: str, actual: str
    ) -> str:
        """Get a more specific subcategory."""
        if category == "REASONING_ERROR":
            if "if" in input_text.lower() and "then" in input_text.lower():
                return "conditional_logic"
            if any(w in input_text.lower() for w in ["all", "some", "every", "no"]):
                return "quantifier_logic"
            return "general_reasoning"

        if category == "KNOWLEDGE_ERROR":
            return "factual_recall"

        if category == "HALLUCINATION":
            return "unsupported_claim"

        return "general"

    def _analyze_root_cause(
        self, category: str, input_text: str, expected: str, actual: str
    ) -> str:
        """Analyze the root cause of the failure."""
        causes = {
            "ARITHMETIC_ERROR": "model attempted mental math instead of using deterministic computation",
            "REASONING_ERROR": "incorrect logical inference or missing premise",
            "KNOWLEDGE_ERROR": "fact not in knowledge base or incorrect retrieval",
            "HALLUCINATION": "model generated unsupported or fabricated information",
            "ENTITY_RESOLUTION_ERROR": "failed to distinguish similar entities",
            "INSTRUCTION_FOLLOWING_ERROR": "did not follow formatting or constraint requirements",
            "RETRIEVAL_ERROR": "failed to find or use relevant information",
            "CONTEXT_LOSS": "lost important information from earlier in the context",
            "OVERCONFIDENCE": "expressed high confidence on incorrect answer",
            "UNDERCONFIDENCE": "expressed low confidence on correct answer",
        }
        return causes.get(category, "unknown root cause")

    def _is_regression(self, category: str, input_text: str) -> bool:
        """Check if this failure is a regression (previously fixed)."""
        input_hash = hashlib.md5(input_text.encode()).hexdigest()[:16]
        for prev in self._failures:
            prev_hash = hashlib.md5(prev.input_text.encode()).hexdigest()[:16]
            if prev_hash == input_hash and prev.fix_applied:
                return True
        return False

    def _create_regression_test(self, record: FailureRecord) -> dict[str, Any]:
        """Create a regression test from a failure record."""
        return {
            "test_id": f"REG-{record.failure_id}",
            "failure_id": record.failure_id,
            "input": record.input_text,
            "expected_output": record.expected_output,
            "category": record.category,
            "created_at": record.timestamp,
            "description": f"Regression test for {record.category}: {record.root_cause}",
        }

    def _save_record(self, record: FailureRecord) -> None:
        """Save a failure record to disk."""
        path = self._storage_dir / f"{record.failure_id}.json"
        data = {
            "failure_id": record.failure_id,
            "timestamp": record.timestamp,
            "input_text": record.input_text,
            "expected_output": record.expected_output,
            "actual_output": record.actual_output,
            "category": record.category,
            "subcategory": record.subcategory,
            "severity": record.severity,
            "task_domain": record.task_domain,
            "confidence": record.confidence,
            "latency_ms": record.latency_ms,
            "is_regression": record.is_regression,
            "root_cause": record.root_cause,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_patterns(self) -> list[FailurePattern]:
        """Analyze failure patterns across all recorded failures."""
        by_category: dict[str, list[FailureRecord]] = {}
        for f in self._failures:
            by_category.setdefault(f.category, []).append(f)

        patterns = []
        for category, failures in sorted(by_category.items(), key=lambda x: -len(x[1])):
            avg_conf = sum(f.confidence for f in failures) / len(failures)
            domains = list(set(f.task_domain for f in failures if f.task_domain))
            root_causes = list(set(f.root_cause for f in failures))
            recommendation = self._generate_recommendation(category, failures)

            patterns.append(FailurePattern(
                category=category,
                count=len(failures),
                avg_confidence=round(avg_conf, 3),
                domains_affected=domains,
                root_causes=root_causes,
                recommendation=recommendation,
            ))

        return patterns

    def _generate_recommendation(
        self, category: str, failures: list[FailureRecord]
    ) -> str:
        """Generate a recommendation for fixing this failure pattern."""
        recommendations = {
            "ARITHMETIC_ERROR": "Route arithmetic tasks to deterministic computation (calculator/Python) instead of neural reasoning.",
            "REASONING_ERROR": "Add more training examples for this reasoning type. Consider adding explicit premise tracking.",
            "KNOWLEDGE_ERROR": "Expand knowledge base for the affected domains. Implement fact verification before stating.",
            "HALLUCINATION": "Add confidence threshold: if below 0.5, say 'I don't know' instead of guessing.",
            "ENTITY_RESOLUTION_ERROR": "Implement entity deduplication with fuzzy matching before entity resolution.",
            "INSTRUCTION_FOLLOWING_ERROR": "Add explicit constraint checking before generating output.",
            "RETRIEVAL_ERROR": "Improve search query generation and source ranking.",
            "OVERCONFIDENCE": "Implement calibration: reduce reported confidence when evidence is weak.",
        }
        return recommendations.get(category, "Investigate root cause and add targeted training data.")

    def summary(self) -> dict[str, Any]:
        """Generate a summary of all failures."""
        if not self._failures:
            return {"total_failures": 0}

        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for f in self._failures:
            by_category[f.category] = by_category.get(f.category, 0) + 1
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

        regressions = sum(1 for f in self._failures if f.is_regression)
        patterns = self.get_patterns()

        return {
            "total_failures": len(self._failures),
            "by_category": by_category,
            "by_severity": by_severity,
            "regressions": regressions,
            "regression_tests_generated": len(self._regression_tests),
            "top_patterns": [
                {"category": p.category, "count": p.count, "recommendation": p.recommendation}
                for p in patterns[:5]
            ],
        }

    @property
    def regression_tests(self) -> list[dict[str, Any]]:
        return list(self._regression_tests)
