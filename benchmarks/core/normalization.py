"""
Prompt Normalization — ensures fair comparison across models.

Implements Section 25 of the benchmark spec:
- Identical task wording
- Identical input
- Identical output requirements
- Identical temperature
- Equivalent token limits
- Equivalent tool access
- Equivalent number of attempts

Never secretly gives Sweep a better prompt.
Never gives competing models intentionally bad prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.core.task import BenchmarkTask


@dataclass
class NormalizedPrompt:
    """A normalized prompt ready for execution."""
    system_message: str
    user_message: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptNormalizer:
    """
    Normalizes prompts across models for fair comparison.

    All models receive:
    - The same system prompt (unless mode-specific)
    - The same user message
    - The same temperature
    - Equivalent token limits
    """

    # Standard system prompts by mode
    SYSTEM_PROMPTS = {
        "raw_model": (
            "You are a precise reasoning system. Answer the following question "
            "using only the information provided. If the information is insufficient, "
            "state that clearly. Do not make assumptions beyond what is given."
        ),
        "tool_augmented": (
            "You are a precise reasoning system with access to tools. "
            "Answer the following question accurately. Use tools when helpful, "
            "but do not make unnecessary tool calls."
        ),
        "full_system": (
            "You are a helpful assistant. Answer the following question accurately "
            "and concisely."
        ),
    }

    def __init__(self, mode: str = "raw_model", seed: int = 42) -> None:
        self._mode = mode
        self._seed = seed

    def normalize(
        self,
        task: BenchmarkTask,
        model_id: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> NormalizedPrompt:
        """
        Create a normalized prompt from a task.

        This ensures all models receive identical inputs.
        """
        system_msg = self.SYSTEM_PROMPTS.get(self._mode, self.SYSTEM_PROMPTS["raw_model"])

        # Build user message with evidence if present
        user_msg = task.query
        if task.evidence:
            evidence_block = "\n".join(f"- {e}" for e in task.evidence)
            user_msg = f"Given the following evidence:\n{evidence_block}\n\nQuestion: {task.query}"

        # Add output format instruction
        if task.evaluation_mode.value == "exact_match":
            user_msg += "\n\nProvide your answer concisely. Just the answer, no explanation unless asked."

        return NormalizedPrompt(
            system_message=system_msg,
            user_message=user_msg,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata={
                "task_id": task.id,
                "category": task.category.value,
                "difficulty": task.difficulty.value,
                "mode": self._mode,
            },
        )

    @staticmethod
    def normalize_temperature(temperature: float) -> float:
        """Ensure temperature is consistent across all models."""
        return 0.0  # Deterministic for benchmark

    @staticmethod
    def normalize_max_tokens(max_tokens: int) -> int:
        """Ensure equivalent token limits."""
        return 1024  # Standard limit for fair comparison

    def get_system_prompt(self, mode: str | None = None) -> str:
        """Get the standard system prompt for a mode."""
        mode = mode or self._mode
        return self.SYSTEM_PROMPTS.get(mode, self.SYSTEM_PROMPTS["raw_model"])
