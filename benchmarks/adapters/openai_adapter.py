"""
OpenAI API Adapter — runs GPT-4o / GPT-4o-mini via the OpenAI API.

Requires OPENAI_API_KEY environment variable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from benchmarks.adapters.base import BaseAdapter
from benchmarks.core.task import BenchmarkTask, TaskResult


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model_id: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("openai package not installed: pip install openai")
        return self._client

    def run(self, task: BenchmarkTask, mode: str = "raw_model") -> TaskResult:
        """Run a benchmark task through OpenAI API."""
        t0 = time.perf_counter()

        try:
            client = self._get_client()

            # Build the prompt
            system_msg = "You are a helpful assistant. Answer the following question accurately."
            user_msg = task.query
            if task.evidence:
                user_msg = f"Evidence:\n" + "\n".join(f"- {e}" for e in task.evidence) + f"\n\nQuestion: {task.query}"

            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            model_answer = response.choices[0].message.content or ""
            tokens_input = response.usage.prompt_tokens if response.usage else 0
            tokens_output = response.usage.completion_tokens if response.usage else 0

        except Exception as e:
            return TaskResult(
                task_id=task.id,
                category=task.category.value,
                subcategory=task.subcategory,
                difficulty=task.difficulty.value,
                model_answer="",
                score=0.0,
                is_correct=False,
                failure_category="INFRASTRUCTURE_FAILURE",
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={"error": str(e)},
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        return TaskResult(
            task_id=task.id,
            category=task.category.value,
            subcategory=task.subcategory,
            difficulty=task.difficulty.value,
            model_answer=model_answer,
            model_reasoning="",
            model_confidence=0.0,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

    def health_check(self) -> bool:
        return bool(self._api_key)
