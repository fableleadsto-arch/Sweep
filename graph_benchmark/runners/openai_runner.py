"""
OpenAI Runner — Runs OpenAI models on graph reasoning tasks.

Requires: OPENAI_API_KEY environment variable.
Uses direct API calls (no agent frameworks).
"""
from __future__ import annotations

import json
import os
import time
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_benchmark.generator.task_generator import Task
from graph_benchmark.scoring.scorer import BenchmarkScorer, TaskResult


class OpenAIGraphRunner:
    """
    Runs OpenAI API on graph reasoning tasks.

    Args:
        model: OpenAI model identifier (e.g., "o1", "gpt-4o").
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._results: list[TaskResult] = []
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def _call_api(self, prompt: str) -> str:
        """Make a single API call."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a precise graph reasoning assistant. Answer exactly as requested. Do not explain."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"

    def _extract_prediction(self, raw_response: str, task: Task) -> str:
        """Extract a structured prediction from the API response."""
        text = raw_response.strip()
        task_type = task.task_type

        if task_type == "reachability":
            m = re.search(r'\b(YES|NO)\b', text, re.IGNORECASE)
            return m.group(1).upper() if m else "NO"

        if task_type == "multi_hop_chain":
            m = re.search(r'\b(YES|NO)\b', text, re.IGNORECASE)
            return m.group(1).upper() if m else "NO"

        if task_type == "contradictory":
            m = re.search(r'\b(CLAIM1|CLAIM2|BOTH|NEITHER)\b', text, re.IGNORECASE)
            return m.group(1).upper() if m else "NEITHER"

        if task_type == "shortest_path":
            m = re.search(r'-?\d+', text)
            return m.group(0) if m else "-1"

        # Set-based: extract comma-separated IDs
        hex_pattern = r'[0-9A-F]{4,8}'
        found = re.findall(hex_pattern, text.upper())
        if found:
            return ", ".join(sorted(set(found)))
        return text

    def run_single(self, task: Task) -> TaskResult:
        """Run a single task."""
        t0 = time.perf_counter()
        raw = self._call_api(task.prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        prediction = self._extract_prediction(raw, task)

        scorer = BenchmarkScorer()
        return scorer.score_task(
            task_id=task.id,
            task_type=task.task_type,
            difficulty=task.difficulty,
            graph_id=task.graph_id,
            ground_truth=task.ground_truth,
            prediction=prediction,
            latency_ms=latency_ms,
            metadata={"model": self._model, "raw_response": raw[:500]},
        )

    def run_all(self, tasks: list[Task], verbose: bool = True) -> dict[str, Any]:
        """Run all tasks."""
        self._results = []
        t_start = time.perf_counter()

        for i, task in enumerate(tasks):
            result = self.run_single(task)
            self._results.append(result)
            if verbose and (i + 1) % 10 == 0:
                print(f"  OpenAI [{self._model}]: {i+1}/{len(tasks)} tasks completed")

        t_total = time.perf_counter() - t_start
        total = len(self._results)
        correct = sum(1 for r in self._results if r.correct)

        return {
            "system": f"openai_{self._model}",
            "model": self._model,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "total_time_s": round(t_total, 2),
            "avg_latency_ms": round(sum(r.latency_ms for r in self._results) / max(1, total), 3),
            "results": self._results,
        }
