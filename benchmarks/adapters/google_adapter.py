"""
Google Gemini Adapter — runs Gemini models via the Google API.

Requires GOOGLE_API_KEY environment variable.
"""
from __future__ import annotations

import os
import time
from typing import Any

from benchmarks.adapters.base import BaseAdapter
from benchmarks.core.task import BenchmarkTask, TaskResult


class GoogleAdapter(BaseAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model_id: str = "gemini-2.5-pro", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")

    def run(self, task: BenchmarkTask, mode: str = "raw_model") -> TaskResult:
        t0 = time.perf_counter()
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(self.model_id)

            prompt = task.query
            if task.evidence:
                prompt = "Evidence:\n" + "\n".join(f"- {e}" for e in task.evidence) + f"\n\nQuestion: {task.query}"

            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            model_answer = response.text or ""
            tokens_input = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            tokens_output = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        except Exception as e:
            return TaskResult(
                task_id=task.id, category=task.category.value,
                subcategory=task.subcategory, difficulty=task.difficulty.value,
                model_answer="", score=0.0, is_correct=False,
                failure_category="INFRASTRUCTURE_FAILURE",
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={"error": str(e)},
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return TaskResult(
            task_id=task.id, category=task.category.value,
            subcategory=task.subcategory, difficulty=task.difficulty.value,
            model_answer=model_answer, model_reasoning="",
            model_confidence=0.0, latency_ms=latency_ms,
            tokens_input=tokens_input, tokens_output=tokens_output,
        )

    def health_check(self) -> bool:
        return bool(self._api_key)
