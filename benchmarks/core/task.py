"""
Benchmark Task Definitions — the fundamental unit of evaluation.

Every task has:
- A unique ID
- A category and subcategory
- Input (query + context/evidence)
- Expected output (ground truth)
- Scoring function
- Metadata (difficulty, contamination status, generation info)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskCategory(Enum):
    """All benchmark task categories."""
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    CODING = "coding"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    LANGUAGE = "language"
    DATA_ANALYSIS = "data_analysis"
    MULTIMODAL = "multimodal"
    RETRIEVAL = "retrieval"
    ENTITY_RESOLUTION = "entity_resolution"
    EVIDENCE_REASONING = "evidence_reasoning"
    MEMORY = "memory"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    WEB_RESEARCH = "web_research"
    UNCERTAINTY = "uncertainty"
    ADVERSARIAL = "adversarial"
    SWEEP_SPECIFIC = "sweep_specific"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationMode(Enum):
    """How to evaluate the task output."""
    EXACT_MATCH = "exact_match"
    CONTAINS_MATCH = "contains_match"
    NUMERIC_MATCH = "numeric_match"
    EXECUTABLE = "executable"
    STRUCTURED = "structured"
    FUZZY_MATCH = "fuzzy_match"
    LLM_JUDGE = "llm_judge"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    category: TaskCategory
    subcategory: str
    query: str
    expected_answer: Any
    evaluation_mode: EvaluationMode
    difficulty: Difficulty = Difficulty.MEDIUM
    evidence: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Contamination control
    dataset_hash: str = ""
    generation_date: str = ""
    source: str = "generated"
    # Confidence requirement
    requires_confidence: bool = False
    # Tool requirements
    requires_tools: bool = False
    # Grouping for multi-part tasks
    group_id: str = ""
    # Hidden test flag
    is_hidden_test: bool = False

    def compute_hash(self) -> str:
        """Compute a hash for contamination tracking."""
        content = json.dumps({
            "id": self.id,
            "query": self.query,
            "expected_answer": str(self.expected_answer),
            "category": self.category.value,
        }, sort_keys=True)
        self.dataset_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.dataset_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "query": self.query,
            "expected_answer": str(self.expected_answer),
            "evaluation_mode": self.evaluation_mode.value,
            "difficulty": self.difficulty.value,
            "evidence": self.evidence,
            "context": self.context,
            "metadata": self.metadata,
            "dataset_hash": self.dataset_hash,
            "source": self.source,
            "requires_confidence": self.requires_confidence,
            "requires_tools": self.requires_tools,
            "group_id": self.group_id,
            "is_hidden_test": self.is_hidden_test,
        }


@dataclass
class TaskResult:
    """Result of executing a single benchmark task."""
    task_id: str
    category: str
    subcategory: str
    difficulty: str
    # Model output
    model_answer: str = ""
    model_reasoning: str = ""
    model_confidence: float = 0.0
    # Scoring
    score: float = 0.0
    max_score: float = 1.0
    is_correct: bool = False
    failure_category: str = ""
    # Timing
    latency_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    # Tool usage
    tool_calls: int = 0
    search_calls: int = 0
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    multi_run_scores: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "difficulty": self.difficulty,
            "model_answer": self.model_answer,
            "model_reasoning": self.model_reasoning,
            "model_confidence": self.model_confidence,
            "score": self.score,
            "max_score": self.max_score,
            "accuracy": self.accuracy,
            "is_correct": self.is_correct,
            "failure_category": self.failure_category,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tool_calls": self.tool_calls,
            "search_calls": self.search_calls,
            "metadata": self.metadata,
            "multi_run_scores": self.multi_run_scores,
        }
