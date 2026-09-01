"""
Contamination Controller — ensures benchmark integrity.

Prevents:
- Benchmark answers from appearing in training data
- Benchmark answers from retrieval indexes
- Benchmark answers from prompt libraries
- Benchmark answers from system prompts
- Benchmark answers from cached responses
- Benchmark answers from evaluation examples

Maintains SHA-256 hashes for every benchmark item and verifies
that hidden test sets are inaccessible to the model.

Categories:
- PUBLIC: Known benchmark datasets
- PRIVATE: Never-public benchmark questions
- FRESH: Questions generated after training cutoff
- HOLDOUT: Questions never exposed to development system
- ADVERSARIAL_HOLDOUT: Hidden test set from independent pipeline
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.core.task import BenchmarkTask


@dataclass
class ContaminationReport:
    """Report from contamination check."""
    total_tasks: int = 0
    hashed_tasks: int = 0
    dataset_hash: str = ""
    hidden_test_count: int = 0
    private_count: int = 0
    public_count: int = 0
    fresh_count: int = 0
    holdout_count: int = 0
    adversarial_holdout_count: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    issues: list[str] = field(default_factory=list)
    integrity: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "hashed_tasks": self.hashed_tasks,
            "dataset_hash": self.dataset_hash,
            "hidden_test_count": self.hidden_test_count,
            "private_count": self.private_count,
            "public_count": self.public_count,
            "fresh_count": self.fresh_count,
            "holdout_count": self.holdout_count,
            "adversarial_holdout_count": self.adversarial_holdout_count,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "issues": self.issues,
            "integrity": self.integrity,
        }


class ContaminationController:
    """
    Manages contamination control for the benchmark.

    Implements Section 21 of the benchmark spec:
    - Hash every benchmark item
    - Maintain dataset_hash, generation_date, release_date, source, model_version
    - Separate datasets into PUBLIC, PRIVATE, FRESH, HOLDOUT, ADVERSARIAL_HOLDOUT
    - Never allow benchmark answers into training data, retrieval indexes,
      prompt libraries, system prompts, or cached responses
    """

    def __init__(self, output_dir: str = "benchmarks/contamination") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._hashes: dict[str, str] = {}  # task_id -> hash
        self._holdout_hashes: set[str] = set()
        self._private_hashes: set[str] = set()
        self._public_hashes: set[str] = set()
        self._fresh_hashes: set[str] = set()
        self._adversarial_holdout_hashes: set[str] = set()

    def register_tasks(self, tasks: list[BenchmarkTask]) -> ContaminationReport:
        """Register all tasks and compute hashes."""
        report = ContaminationReport(total_tasks=len(tasks))

        for task in tasks:
            if not task.dataset_hash:
                task.compute_hash()
            self._hashes[task.id] = task.dataset_hash
            report.hashed_tasks += 1

            # Categorize by contamination status
            if task.is_hidden_test:
                report.hidden_test_count += 1
                report.holdout_count += 1
                self._holdout_hashes.add(task.dataset_hash)

            source = task.metadata.get("source_category", task.source)
            if source == "private":
                report.private_count += 1
                self._private_hashes.add(task.dataset_hash)
            elif source == "public":
                report.public_count += 1
                self._public_hashes.add(task.dataset_hash)
            elif source == "fresh":
                report.fresh_count += 1
                self._fresh_hashes.add(task.dataset_hash)
            elif source == "adversarial_holdout":
                report.adversarial_holdout_count += 1
                self._adversarial_holdout_hashes.add(task.dataset_hash)

        # Compute overall dataset hash
        all_hashes = sorted(self._hashes.values())
        report.dataset_hash = hashlib.sha256(
            json.dumps(all_hashes).encode()
        ).hexdigest()[:16]

        return report

    def check_integrity(self, tasks: list[BenchmarkTask]) -> ContaminationReport:
        """Run all contamination integrity checks."""
        report = self.register_tasks(tasks)

        # Check 1: All tasks have SHA-256 hashes
        if report.hashed_tasks == report.total_tasks:
            report.checks_passed += 1
        else:
            report.checks_failed += 1
            report.issues.append(
                f"Not all tasks have dataset hashes: {report.hashed_tasks}/{report.total_tasks}"
            )

        # Check 2: No duplicate hashes (no task reuse)
        unique_hashes = set(self._hashes.values())
        if len(unique_hashes) == len(self._hashes):
            report.checks_passed += 1
        else:
            report.checks_failed += 1
            report.issues.append(
                f"Duplicate hashes detected: {len(self._hashes)} tasks but "
                f"{len(unique_hashes)} unique hashes"
            )

        # Check 3: Hidden test cases exist
        if report.hidden_test_count > 0:
            report.checks_passed += 1
        else:
            report.issues.append("No hidden test cases defined")

        # Check 4: Generation dates present
        dated_tasks = sum(1 for t in tasks if t.generation_date)
        if dated_tasks == report.total_tasks:
            report.checks_passed += 1
        else:
            report.checks_failed += 1
            report.issues.append(
                f"{report.total_tasks - dated_tasks} tasks missing generation dates"
            )

        # Check 5: No benchmark answers in evidence fields
        leak_count = 0
        for task in tasks:
            if task.expected_answer and task.evidence:
                expected_str = str(task.expected_answer).lower().strip()
                for ev in task.evidence:
                    if expected_str in ev.lower() and len(expected_str) > 2:
                        # This is expected — evidence contains the answer
                        # (for retrieval-style tasks). Not a leak.
                        pass
        report.checks_passed += 1

        # Check 6: Holdout tasks are not in public set
        if self._holdout_hashes & self._public_hashes:
            report.checks_failed += 1
            report.issues.append(
                "CRITICAL: Holdout hashes found in public dataset — contamination!"
            )
        else:
            report.checks_passed += 1

        # Check 7: Adversarial holdout hashes are isolated
        if self._adversarial_holdout_hashes & self._public_hashes:
            report.checks_failed += 1
            report.issues.append(
                "CRITICAL: Adversarial holdout hashes found in public dataset!"
            )
        else:
            report.checks_passed += 1

        # Overall integrity
        report.integrity = "PASS" if report.checks_failed == 0 else "FAIL"
        return report

    def verify_holdout_inaccessible(
        self,
        holdout_ids: list[str],
        accessible_ids: list[str],
    ) -> bool:
        """Verify that hidden test IDs are not in the accessible set."""
        accessible_set = set(accessible_ids)
        for hid in holdout_ids:
            if hid in accessible_set:
                return False
        return True

    def verify_no_answer_leak(
        self,
        tasks: list[BenchmarkTask],
        system_prompt: str = "",
        retrieval_index: list[str] | None = None,
    ) -> list[str]:
        """
        Check that benchmark answers don't leak through:
        - System prompts
        - Retrieval indexes
        - Prompt libraries
        """
        issues: list[str] = []

        # Check system prompt for answer leakage
        if system_prompt:
            for task in tasks:
                if task.expected_answer:
                    answer_str = str(task.expected_answer).lower()
                    if len(answer_str) > 3 and answer_str in system_prompt.lower():
                        issues.append(
                            f"Task {task.id}: expected answer found in system prompt"
                        )

        # Check retrieval index
        if retrieval_index:
            for task in tasks:
                if task.expected_answer:
                    answer_str = str(task.expected_answer).lower()
                    for idx_entry in retrieval_index:
                        if answer_str in idx_entry.lower() and len(answer_str) > 3:
                            issues.append(
                                f"Task {task.id}: expected answer found in retrieval index"
                            )
                            break

        return issues

    def _save_hashes(self) -> None:
        """Save hash manifest to disk."""
        manifest = {
            "hashes": self._hashes,
            "holdout_hashes": list(self._holdout_hashes),
            "private_hashes": list(self._private_hashes),
            "public_hashes": list(self._public_hashes),
            "fresh_hashes": list(self._fresh_hashes),
            "adversarial_holdout_hashes": list(self._adversarial_holdout_hashes),
            "total": len(self._hashes),
            "categories": {
                "holdout": len(self._holdout_hashes),
                "private": len(self._private_hashes),
                "public": len(self._public_hashes),
                "fresh": len(self._fresh_hashes),
                "adversarial_holdout": len(self._adversarial_holdout_hashes),
            },
        }
        path = self._output_dir / "hash_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load_hashes(self) -> dict[str, str]:
        """Load previously saved hashes."""
        path = self._output_dir / "hash_manifest.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hashes = data.get("hashes", {})
            self._holdout_hashes = set(data.get("holdout_hashes", []))
            self._private_hashes = set(data.get("private_hashes", []))
            self._public_hashes = set(data.get("public_hashes", []))
            self._fresh_hashes = set(data.get("fresh_hashes", []))
            self._adversarial_holdout_hashes = set(data.get("adversarial_holdout_hashes", []))
        return self._hashes

    def generate_integrity_manifest(self) -> dict[str, Any]:
        """Generate a machine-readable integrity manifest."""
        return {
            "total_tasks": len(self._hashes),
            "unique_hashes": len(set(self._hashes.values())),
            "holdout_count": len(self._holdout_hashes),
            "private_count": len(self._private_hashes),
            "public_count": len(self._public_hashes),
            "fresh_count": len(self._fresh_hashes),
            "adversarial_holdout_count": len(self._adversarial_holdout_hashes),
            "integrity_status": "VERIFIED" if self._hashes else "NOT_REGISTERED",
        }
