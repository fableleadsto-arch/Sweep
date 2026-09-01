"""
Base Adapter — abstract interface for all model adapters.

Every adapter must implement run() to execute a single benchmark task.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmarks.core.task import BenchmarkTask, TaskResult


class BaseAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.config = kwargs

    @abstractmethod
    def run(self, task: BenchmarkTask, mode: str = "raw_model") -> TaskResult:
        """
        Execute a single benchmark task.

        Args:
            task: The benchmark task to execute
            mode: Comparison mode (raw_model, tool_augmented, full_system)

        Returns:
            TaskResult with the model's answer and metadata
        """
        ...

    def health_check(self) -> bool:
        """Check if the adapter is available and configured."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id})"
