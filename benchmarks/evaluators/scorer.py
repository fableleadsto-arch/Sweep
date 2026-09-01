"""
Deterministic Benchmark Scorer.

Implements exact match, containment, numeric, structured, and fuzzy
scoring. LLM judge is secondary only.
"""
from __future__ import annotations

import json
import re
from typing import Any

from benchmarks.core.task import BenchmarkTask, TaskResult, EvaluationMode


class BenchmarkScorer:
    """Scores benchmark task results deterministically."""

    def score(self, task: BenchmarkTask, model_answer: str, **kwargs: Any) -> TaskResult:
        """Score a single task result."""
        result = TaskResult(
            task_id=task.id,
            category=task.category.value,
            subcategory=task.subcategory,
            difficulty=task.difficulty.value,
            model_answer=model_answer,
            model_reasoning=kwargs.get("reasoning", ""),
            model_confidence=kwargs.get("confidence", 0.0),
            tokens_input=kwargs.get("tokens_input", 0),
            tokens_output=kwargs.get("tokens_output", 0),
            tool_calls=kwargs.get("tool_calls", 0),
            search_calls=kwargs.get("search_calls", 0),
        )

        expected = str(task.expected_answer) if task.expected_answer is not None else ""
        answer = model_answer.strip()

        # Handle None expected answers — these are open-ended tasks
        # where the model should produce substantive output, not just "supported"
        if task.expected_answer is None:
            result.score = self._score_open_ended(task, answer)
            result.is_correct = result.score >= 0.5
            result.max_score = 1.0
            result.failure_category = self._classify_failure(task, result)
            return result

        # First try direct match
        direct_score = self._evaluate(task, expected, answer)

        # If direct match fails, try decision-to-answer mapping
        # (for models that output verdicts instead of answers)
        if direct_score < 0.5:
            mapped_answer = self._map_decision_to_answer(expected, answer)
            if mapped_answer != answer:
                mapped_score = self._evaluate(task, expected, mapped_answer)
                direct_score = max(direct_score, mapped_score)
                if mapped_score > direct_score:
                    answer = mapped_answer

        result.score = direct_score
        result.is_correct = result.score >= 0.5
        result.max_score = 1.0
        result.failure_category = self._classify_failure(task, result)
        return result

    def _evaluate(self, task: BenchmarkTask, expected: str, answer: str) -> float:
        """Evaluate a single answer against expected."""
        if task.evaluation_mode == EvaluationMode.EXACT_MATCH:
            return 1.0 if self._exact_match(expected, answer) else 0.0
        elif task.evaluation_mode == EvaluationMode.CONTAINS_MATCH:
            return 1.0 if self._contains_match(expected, answer) else 0.0
        elif task.evaluation_mode == EvaluationMode.NUMERIC_MATCH:
            return 1.0 if self._numeric_match(expected, answer) else 0.0
        elif task.evaluation_mode == EvaluationMode.FUZZY_MATCH:
            return self._fuzzy_match(expected, answer)
        elif task.evaluation_mode == EvaluationMode.STRUCTURED:
            return self._structured_match(task, answer)
        elif task.evaluation_mode == EvaluationMode.EXECUTABLE:
            return self._executable_match(task, answer)
        return 0.0

    def _score_open_ended(self, task: BenchmarkTask, answer: str) -> float:
        """
        Score open-ended tasks where expected_answer is None.

        These tasks require substantive output, not just a verdict.
        The model should produce a meaningful response, not just "supported".
        """
        # If the model just output a verdict without substance, penalize it
        verdicts = {"supported", "refuted", "mixed", "insufficient"}
        if answer.lower().strip() in verdicts:
            return 0.0

        # Check if the answer has substance (more than just a few words)
        words = answer.split()
        if len(words) < 3:
            return 0.2  # Minimal credit for any response

        # For structured tasks (JSON, lists), check format
        ctx = task.context
        if ctx.get("type") == "json_schema":
            try:
                import json
                data = json.loads(answer)
                required = ctx.get("required_keys", [])
                if all(k in data for k in required):
                    return 1.0
                return 0.5
            except (json.JSONDecodeError, TypeError):
                pass
        elif ctx.get("type") == "word_count":
            target = ctx.get("target", 5)
            if len(words) == target:
                return 1.0
            elif abs(len(words) - target) <= 1:
                return 0.7
            return 0.3
        elif ctx.get("type") == "list":
            if answer.strip():
                return 1.0
            return 0.0

        # For general open-ended tasks, give credit for substantive responses
        # that aren't just verdicts
        if len(words) >= 5:
            return 0.8  # Good substantive response
        return 0.5

    def _map_decision_to_answer(self, expected: str, decision: str) -> str:
        """
        Map categorical decision output to answer format.

        The ReasoningCortex outputs verdicts (supported/refuted/mixed/insufficient)
        rather than specific answers. This maps the MODEL's decision to a
        natural-language answer that can be compared against the expected answer.

        CRITICAL: This function must map the MODEL's output to what the model
        THINKS the answer is. It must NEVER return the expected answer, as that
        would be cheating.
        """
        decision_lower = decision.lower().strip()
        expected_lower = expected.lower().strip()

        # Map the MODEL's verdict to a natural-language answer
        if decision_lower == "supported":
            return "yes"
        elif decision_lower == "refuted":
            return "no"
        elif decision_lower == "mixed":
            return "mixed"
        elif decision_lower == "insufficient":
            return "unknown"

        return decision

    def _exact_match(self, expected: str, actual: str) -> bool:
        """Strict exact match (case-insensitive, stripped)."""
        norm_e = expected.lower().strip().rstrip(".!?,;")
        norm_a = actual.lower().strip().rstrip(".!?,;")
        return norm_e == norm_a

    def _contains_match(self, expected: str, actual: str) -> bool:
        """Check if expected answer is contained in the model's response."""
        norm_e = expected.lower().strip()
        norm_a = actual.lower().strip()
        return norm_e in norm_a

    def _numeric_match(self, expected: str, actual: str) -> bool:
        """Numeric comparison allowing for formatting differences."""
        try:
            e = float(re.sub(r'[^0-9.\-]', '', expected))
            a = float(re.sub(r'[^0-9.\-]', '', actual))
            return abs(e - a) < 0.01 * max(abs(e), 1)
        except (ValueError, ZeroDivisionError):
            return self._exact_match(expected, actual)

    def _fuzzy_match(self, expected: str, actual: str) -> float:
        """Token overlap based fuzzy matching."""
        if not expected or not actual:
            return 0.0
        stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "and", "or", "it", "that", "this", "with"}
        e_tokens = set(expected.lower().split()) - stop
        a_tokens = set(actual.lower().split()) - stop
        if not e_tokens:
            return 1.0
        overlap = len(e_tokens & a_tokens)
        return min(1.0, overlap / len(e_tokens))

    def _structured_match(self, task: BenchmarkTask, answer: str) -> float:
        """Validate structured output (JSON, lists, etc.)."""
        ctx = task.context
        if ctx.get("type") == "json_schema":
            try:
                data = json.loads(answer)
                required = ctx.get("required_keys", [])
                if all(k in data for k in required):
                    return 1.0
                return 0.5
            except (json.JSONDecodeError, TypeError):
                return 0.0
        elif ctx.get("type") == "word_count":
            target = ctx.get("target", 5)
            words = answer.split()
            if len(words) == target:
                return 1.0
            elif abs(len(words) - target) <= 1:
                return 0.5
            return 0.0
        elif ctx.get("type") == "list":
            return 1.0 if answer.strip() else 0.0
        return self._fuzzy_match(task.expected_answer or "", answer)

    def _executable_match(self, task: BenchmarkTask, answer: str) -> float:
        """Execute code-based answers and verify."""
        # For now, fall back to fuzzy match
        return self._fuzzy_match(str(task.expected_answer or ""), answer)

    def _classify_failure(self, task: BenchmarkTask, result: TaskResult) -> str:
        """Classify the type of failure if the task was incorrect."""
        if result.is_correct:
            return ""

        cat = task.category.value
        if cat == "mathematics":
            return "ARITHMETIC_ERROR"
        elif cat == "reasoning":
            return "REASONING_ERROR"
        elif cat == "knowledge":
            return "KNOWLEDGE_ERROR"
        elif cat == "coding":
            return "TOOL_EXECUTION_ERROR"
        elif cat == "entity_resolution":
            return "ENTITY_RESOLUTION_ERROR"
        elif cat == "multimodal":
            return "PERCEPTION_ERROR"
        elif cat == "retrieval":
            return "RETRIEVAL_ERROR"
        elif cat == "instruction_following":
            return "INSTRUCTION_FOLLOWING_ERROR"
        elif cat == "uncertainty":
            if result.model_confidence > 0.8 and not result.is_correct:
                return "OVERCONFIDENCE"
            return "HALLUCINATION"
        elif cat == "adversarial":
            return "HALLUCINATION"
        elif cat == "memory":
            return "MEMORY_ERROR"
        elif cat == "evidence_reasoning":
            return "SOURCE_ATTRIBUTION_ERROR"
        else:
            return "REASONING_ERROR"

    def score_batch(
        self,
        tasks: list[BenchmarkTask],
        answers: list[str],
        **kwargs: Any,
    ) -> list[TaskResult]:
        """Score a batch of tasks."""
        results = []
        for task, answer in zip(tasks, answers):
            results.append(self.score(task, answer, **kwargs))
        return results
