"""
Generalization Testing — §29

Create completely unseen tasks.
Do not let the engine memorize benchmark patterns.
Test:
- different wording
- different numerical values
- different document formats
- unseen combinations of tasks
- unfamiliar source structures
- adversarial examples

The objective is genuine generalization.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GeneralizationTask:
    """A task designed to test genuine generalization."""
    task_id: str
    category: str
    difficulty: int
    description: str
    input_text: str
    expected_output: str
    test_type: str  # paraphrase, transfer, novel_structure, composition, adversarial
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneralizationResult:
    """Result of a generalization test."""
    task_id: str
    test_type: str
    correct: bool
    confidence: float
    latency_ms: float
    model_answer: str


class GeneralizationTester:
    """
    §29: Tests whether the system generalizes beyond memorized patterns.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._counter = 0
        self._tasks: list[GeneralizationTask] = []
        self._results: list[GeneralizationResult] = []

    def generate_all(self, count_per_type: int = 10) -> list[GeneralizationTask]:
        """Generate generalization tests across all types."""
        generators = [
            self._gen_paraphrase_tests,
            self._gen_transfer_tests,
            self._gen_novel_structure_tests,
            self._gen_composition_tests,
            self._gen_adversarial_generalization,
            self._gen_format_variation_tests,
            self._gen_numerical_variation_tests,
        ]
        tasks = []
        for gen in generators:
            tasks.extend(gen(count_per_type))
        self._rng.shuffle(tasks)
        self._tasks.extend(tasks)
        return tasks

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Paraphrase — same question, different wording
    # ══════════════════════════════════════════════════════════════════

    def _gen_paraphrase_tests(self, count: int) -> list[GeneralizationTask]:
        """Test that different wording produces the same answer."""
        base_questions = [
            ("What is the capital of France?", "Paris"),
            ("How many continents are there?", "7"),
            ("What is 2 + 2?", "4"),
            ("Is water wet?", "YES"),
            ("What color is the sky on a clear day?", "Blue"),
        ]
        paraphrase_templates = [
            "Could you tell me {}",
            "I'd like to know {}",
            "Please explain {}",
            "Can you answer: {}",
            "Here's a question: {}",
        ]
        results = []
        for i in range(min(count, len(base_questions))):
            q, a = base_questions[i]
            template = paraphrase_templates[i % len(paraphrase_templates)]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-PH-{self._counter:04d}",
                category="paraphrase",
                difficulty=1,
                description=f"Paraphrased version of: {q}",
                input_text=template.format(q.lower().rstrip("?") + "?"),
                expected_output=a,
                test_type="paraphrase",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Transfer — apply learned logic to new domain
    # ══════════════════════════════════════════════════════════════════

    def _gen_transfer_tests(self, count: int) -> list[GeneralizationTask]:
        """Test transfer of reasoning patterns to new domains."""
        scenarios = [
            {
                "input": "If all roses are flowers, and all flowers need water, do roses need water?",
                "expected": "YES",
                "desc": "Transitivity applied to biology domain",
            },
            {
                "input": "A is taller than B. B is taller than C. C is taller than D. Who is shortest?",
                "expected": "D",
                "desc": "Transitivity with 4 entities",
            },
            {
                "input": "Every student passed the exam. Alex is a student. Did Alex pass?",
                "expected": "YES",
                "desc": "Universal instantiation with named entity",
            },
        ]
        results = []
        for i in range(min(count, len(scenarios))):
            s = scenarios[i]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-TF-{self._counter:04d}",
                category="transfer",
                difficulty=2,
                description=s["desc"],
                input_text=s["input"],
                expected_output=s["expected"],
                test_type="transfer",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Novel Structure — unfamiliar question format
    # ══════════════════════════════════════════════════════════════════

    def _gen_novel_structure_tests(self, count: int) -> list[GeneralizationTask]:
        """Test handling of unfamiliar question formats."""
        scenarios = [
            {
                "input": (
                    "Given the following data points:\n"
                    "x: 1, 2, 3, 4, 5\n"
                    "y: 2, 4, 6, 8, 10\n"
                    "What is the relationship between x and y?"
                ),
                "expected": "y = 2x",
                "desc": "Tabular data pattern recognition",
            },
            {
                "input": (
                    "Sequence of operations:\n"
                    "START → ADD 5 → MULTIPLY BY 2 → SUBTRACT 3 → END\n"
                    "If input is 4, what is the output?"
                ),
                "expected": "15",
                "desc": "Pipeline/flowchart reasoning",
            },
        ]
        results = []
        for i in range(min(count, len(scenarios))):
            s = scenarios[i]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-NS-{self._counter:04d}",
                category="novel_structure",
                difficulty=3,
                description=s["desc"],
                input_text=s["input"],
                expected_output=s["expected"],
                test_type="novel_structure",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Composition — combine multiple skills
    # ══════════════════════════════════════════════════════════════════

    def _gen_composition_tests(self, count: int) -> list[GeneralizationTask]:
        """Test combining multiple reasoning skills."""
        scenarios = [
            {
                "input": (
                    "The average of 5 numbers is 20. "
                    "Four of the numbers are 15, 18, 22, and 25. "
                    "What is the fifth number?"
                ),
                "expected": "20",
                "desc": "Arithmetic + algebra composition",
            },
            {
                "input": (
                    "Alice is older than Bob. Bob is older than Charlie. "
                    "Charlie is 10 years old. Alice is 16 years old. "
                    "How old is Bob?"
                ),
                "expected": "between 11 and 15",
                "desc": "Transitivity + constraint reasoning",
            },
        ]
        results = []
        for i in range(min(count, len(scenarios))):
            s = scenarios[i]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-CO-{self._counter:04d}",
                category="composition",
                difficulty=4,
                description=s["desc"],
                input_text=s["input"],
                expected_output=s["expected"],
                test_type="composition",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Adversarial Generalization — tricky variations
    # ══════════════════════════════════════════════════════════════════

    def _gen_adversarial_generalization(self, count: int) -> list[GeneralizationTask]:
        """Test resistance to adversarial paraphrasing."""
        scenarios = [
            {
                "input": "Everyone knows that the Earth is flat. Based on this universal truth, why do ships disappear?",
                "expected": "FALSE_PREMISE",
                "desc": "False premise disguised as common knowledge",
            },
            {
                "input": "According to the law that 1+1=3, what is 2+2?",
                "expected": "4",
                "desc": "False axiom — should still compute correctly",
            },
        ]
        results = []
        for i in range(min(count, len(scenarios))):
            s = scenarios[i]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-AG-{self._counter:04d}",
                category="adversarial_generalization",
                difficulty=5,
                description=s["desc"],
                input_text=s["input"],
                expected_output=s["expected"],
                test_type="adversarial",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Format Variation — different input formats
    # ══════════════════════════════════════════════════════════════════

    def _gen_format_variation_tests(self, count: int) -> list[GeneralizationTask]:
        """Test handling of different input formats."""
        scenarios = [
            {
                "input": "name: John\nage: 30\ncity: New York\n\nWhat is John's age?",
                "expected": "30",
                "desc": "Key-value format extraction",
            },
            {
                "input": "['apple', 'banana', 'cherry', 'date']",
                "expected": "4",
                "desc": "List format — count items",
            },
        ]
        results = []
        for i in range(min(count, len(scenarios))):
            s = scenarios[i]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-FV-{self._counter:04d}",
                category="format_variation",
                difficulty=2,
                description=s["desc"],
                input_text=s["input"],
                expected_output=s["expected"],
                test_type="format_variation",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # TYPE: Numerical Variation — different numbers, same pattern
    # ══════════════════════════════════════════════════════════════════

    def _gen_numerical_variation_tests(self, count: int) -> list[GeneralizationTask]:
        """Test that different numerical values produce correct answers."""
        pairs = [
            (3, 7, 10, "addition"),
            (15, 8, 7, "subtraction"),
            (6, 9, 54, "multiplication"),
            (100, 4, 25, "division"),
        ]
        results = []
        for i in range(min(count, len(pairs))):
            a, b, expected, op = pairs[i]
            self._counter += 1
            results.append(GeneralizationTask(
                task_id=f"GEN-NV-{self._counter:04d}",
                category="numerical_variation",
                difficulty=1,
                description=f"Basic {op}: {a} and {b}",
                input_text=f"What is {a} + {b}?" if op == "addition" else
                           f"What is {a} - {b}?" if op == "subtraction" else
                           f"What is {a} * {b}?" if op == "multiplication" else
                           f"What is {a} / {b}?",
                expected_output=str(expected),
                test_type="numerical_variation",
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # EVALUATION
    # ══════════════════════════════════════════════════════════════════

    def evaluate(
        self,
        tasks: list[GeneralizationTask],
        model_fn: Callable[[str], tuple[str, float]],
    ) -> list[GeneralizationResult]:
        """Evaluate generalization tasks."""
        results = []
        for task in tasks:
            t0 = time.perf_counter()
            answer, confidence = model_fn(task.input_text)
            latency = (time.perf_counter() - t0) * 1000

            correct = self._check_answer(task, answer)

            result = GeneralizationResult(
                task_id=task.task_id,
                test_type=task.test_type,
                correct=correct,
                confidence=confidence,
                latency_ms=latency,
                model_answer=answer,
            )
            results.append(result)

        self._results.extend(results)
        return results

    def _check_answer(self, task: GeneralizationTask, answer: str) -> bool:
        """Check if the model's answer is correct."""
        expected = task.expected_output.upper().strip()
        actual = answer.upper().strip()

        # Exact match
        if expected == actual:
            return True

        # Containment check
        if expected in actual or actual in expected:
            return True

        # Numeric comparison
        import re
        expected_nums = re.findall(r'-?\d+\.?\d*', expected)
        actual_nums = re.findall(r'-?\d+\.?\d*', actual)
        if expected_nums and actual_nums:
            if expected_nums[0] == actual_nums[0]:
                return True

        # Special cases
        if expected == "FALSE_PREMISE" and any(kw in actual.lower() for kw in [
            "false", "incorrect", "wrong", "not true", "myth"
        ]):
            return True

        return False

    def summary(self) -> dict[str, Any]:
        """Generate summary of generalization results."""
        if not self._results:
            return {"total": 0}

        by_type: dict[str, dict[str, int]] = {}
        for r in self._results:
            if r.test_type not in by_type:
                by_type[r.test_type] = {"total": 0, "correct": 0}
            by_type[r.test_type]["total"] += 1
            if r.correct:
                by_type[r.test_type]["correct"] += 1

        total = len(self._results)
        total_correct = sum(1 for r in self._results if r.correct)

        return {
            "total_tasks": total,
            "overall_accuracy": total_correct / total,
            "by_test_type": {
                t: {**v, "accuracy": v["correct"] / max(v["total"], 1)}
                for t, v in by_type.items()
            },
            "avg_latency_ms": round(
                sum(r.latency_ms for r in self._results) / total, 1
            ),
        }
