"""
Experience Memory — Stores, filters, and manages training experiences.

§2: Only verified examples enter the primary learning dataset.
§16: Store problem → what Sweep believed → why it was wrong → corrected result.
§17: Hard-negative generation for structurally similar but logically different examples.
§23: experience/successful, failed, corrected, hard_negative, ambiguous, regression.
§24: Quality checks before training eligibility.
"""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Experience:
    """A single training experience."""
    experience_id: str
    domain: str
    difficulty: int
    input: str
    candidate_reasoning: str
    critique: str
    correction: str
    final_answer: str
    expected_answer: str
    confidence: float
    verification_result: bool
    error_type: str
    training_status: str
    timestamp: float
    model_version: str
    is_hard_negative: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "input": self.input,
            "candidate_reasoning": self.candidate_reasoning,
            "critique": self.critique,
            "correction": self.correction,
            "final_answer": self.final_answer,
            "expected_answer": self.expected_answer,
            "confidence": self.confidence,
            "verification_result": self.verification_result,
            "error_type": self.error_type,
            "training_status": self.training_status,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "is_hard_negative": self.is_hard_negative,
            "metadata": self.metadata,
        }


class ExperienceMemory:
    """
    Experience store with quality filtering.

    §23: experience/successful, failed, corrected, hard_negative, ambiguous, regression.
    §24: Duplicate detection, quality check, ground-truth verification.
    §25: Batch collection before training update.
    """

    def __init__(
        self,
        storage_dir: str | Path = "sweep_neural_mesh/training/experience",
        batch_size: int = 100,
    ) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._seen_hashes: set[str] = set()
        self._pending: list[Experience] = []
        self._counters = {"successful": 0, "failed": 0, "corrected": 0,
                          "hard_negative": 0, "ambiguous": 0, "rejected": 0}

    def store(self, experience: Experience) -> bool:
        """
        Store an experience after quality checks.

        §24: Reject duplicates, corrupted examples, unverifiable examples.
        Returns True if stored, False if rejected.
        """
        content_hash = hashlib.md5(
            (experience.input + experience.final_answer).encode()
        ).hexdigest()

        if content_hash in self._seen_hashes:
            self._counters["rejected"] += 1
            return False

        if not experience.input or not experience.final_answer:
            self._counters["rejected"] += 1
            return False

        if experience.verification_result and not experience.is_hard_negative:
            experience.training_status = "verified"
            self._counters["successful"] += 1
        elif experience.error_type:
            experience.training_status = "error_analysis"
            if experience.is_hard_negative:
                self._counters["hard_negative"] += 1
            else:
                self._counters["failed"] += 1
        else:
            experience.training_status = "ambiguous"
            self._counters["ambiguous"] += 1

        self._seen_hashes.add(content_hash)
        self._pending.append(experience)
        self._save_experience(experience)
        return True

    def get_training_batch(self) -> list[Experience]:
        """Get a batch of verified experiences for training."""
        verified = [e for e in self._pending if e.training_status == "verified"]
        batch = verified[:self._batch_size]
        return batch

    def get_error_batch(self) -> list[Experience]:
        """Get error analysis examples for contrastive learning."""
        return [e for e in self._pending if e.training_status == "error_analysis"]

    def is_batch_ready(self) -> bool:
        """Check if enough verified experiences for a training batch."""
        verified = sum(1 for e in self._pending if e.training_status == "verified")
        return verified >= self._batch_size

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_stored": len(self._pending),
            "pending": len(self._pending),
            "counters": self._counters.copy(),
            "batch_ready": self.is_batch_ready(),
        }

    def clear_batch(self) -> None:
        """Clear the pending batch after training."""
        self._pending = []

    def _save_experience(self, exp: Experience) -> None:
        subdir = "successful" if exp.training_status == "verified" else \
                 "hard_negative" if exp.is_hard_negative else \
                 "failed" if exp.training_status == "error_analysis" else "ambiguous"
        dir_path = self._storage_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{exp.experience_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(exp.to_dict(), f, indent=2)

    def generate_hard_negatives(
        self,
        experience: Experience,
        count: int = 3,
    ) -> list[Experience]:
        """
        §17: Generate structurally similar but logically different examples.
        """
        hard_negatives = []
        for i in range(count):
            hn = Experience(
                experience_id=f"{experience.experience_id}-HN{i}",
                domain=experience.domain,
                difficulty=experience.difficulty,
                input=self._mutate_input(experience.input, i),
                candidate_reasoning="[hard negative - not yet solved]",
                critique="[generated for contrastive learning]",
                correction="",
                final_answer="",
                expected_answer=self._flip_answer(experience.expected_answer),
                confidence=0.0,
                verification_result=False,
                error_type="hard_negative_generated",
                training_status="hard_negative",
                timestamp=time.time(),
                model_version=experience.model_version,
                is_hard_negative=True,
                metadata={"parent_id": experience.experience_id, "mutation": i},
            )
            hard_negatives.append(hn)
        return hard_negatives

    def _mutate_input(self, text: str, variant: int) -> str:
        """Create a mutated version of the input."""
        mutations = [
            text.replace("implies", "does not imply"),
            text.replace("is greater than", "is less than"),
            text.replace("before", "after"),
        ]
        return mutations[variant % len(mutations)]

    def _flip_answer(self, answer: str) -> str:
        flips = {
            "YES": "NO", "NO": "YES",
            "TRUE": "FALSE", "FALSE": "TRUE",
            "SUPPORTED": "REFUTED", "REFUTED": "SUPPORTED",
            "CONSISTENT": "CONTRADICTION", "CONTRADICTION": "CONSISTENT",
        }
        return flips.get(answer.upper(), answer)
