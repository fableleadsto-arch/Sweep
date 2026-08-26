"""
Trainer — Core training loop orchestrator.

§1: No general learning rule must exist. Training is explicit.
§2: Primary learning dataset composed of expert-generated verified examples.
§3: Training mechanics cannot be unlearned or degraded through dataset expansion.
§4: Initial parameters set by data analysis, confirmed by benchmark.
§5: Freeze parameters confirmed by benchmark test, use for reasoning.
§6: No online training. Single sweep pass, no cross-domain generalization.
§7: Heavy precomputation and factual cache. (not applicable to training loop)
§8: No updates to unexamined data.
§9: The ability to reason is MORE IMPORTANT than specific facts.
§33: Benchmarks + tests = necessary and sufficient.
§34: Autonomous training mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweep_neural_mesh.training.domains import ExpertiseTracker
from sweep_neural_mesh.training.task_generator import TaskGenerator, Task
from sweep_neural_mesh.training.solver import Solver, SolveResult
from sweep_neural_mesh.training.critique import Critique
from sweep_neural_mesh.training.verifier import Verifier
from sweep_neural_mesh.training.experience import Experience, ExperienceMemory
from sweep_neural_mesh.training.curriculum import CurriculumManager
from sweep_neural_mesh.training.calibration import ConfidenceCalibrator
from sweep_neural_mesh.training.versioning import VersionManager
from sweep_neural_mesh.training.dashboard import Dashboard


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    domains: list[str] = field(default_factory=lambda: [
        "basic_logic", "novel_structures", "ambiguity",
    ])
    tasks_per_domain: int = 1000
    difficulty_level: int = 2
    num_candidates: int = 3
    regression_suite_size: int = 20
    batch_size: int = 100
    max_iterations: int = 100
    mastery_threshold: float = 0.90
    output_dir: str = "sweep_neural_mesh/training/results"


@dataclass
class TrainingResult:
    """Result of a training run."""
    config: TrainingConfig
    duration_seconds: float
    total_tasks_generated: int
    total_tasks_solved: int
    total_correct: int
    accuracy_before: float
    accuracy_after: float
    domain_scores_before: dict[str, float]
    domain_scores_after: dict[str, float]
    version: str
    dashboard_report: dict[str, Any]
    regression_result: dict[str, Any]
    calibration_summary: dict[str, Any]
    experiences_stored: int
    hard_negatives_generated: int


class Trainer:
    """
    §1: Training is explicit, not a general learning rule.
    §2: Verified examples enter primary dataset.
    §3: Mechanics cannot be degraded.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self._config = config or TrainingConfig()
        self._expertise = ExpertiseTracker()
        self._task_gen = TaskGenerator()
        self._solver = Solver(num_candidates=self._config.num_candidates)
        self._critique = Critique()
        self._verifier = Verifier()
        self._experience = ExperienceMemory(batch_size=self._config.batch_size)
        self._curriculum = CurriculumManager(
            expertise=self._expertise,
            mastery_threshold=self._config.mastery_threshold,
            revisit_interval=5,
            regression_suite_size=self._config.regression_suite_size,
        )
        self._calibrator = ConfidenceCalibrator()
        self._version_manager = VersionManager()
        self._dashboard = Dashboard(
            expertise=self._expertise,
            calibrator=self._calibrator,
            version_manager=self._version_manager,
        )
        self._output_dir = Path(self._config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> TrainingResult:
        """Run the full training loop."""
        t0 = time.perf_counter()
        total_generated = 0
        total_solved = 0
        total_correct = 0
        hard_negatives = 0

        snapshot_before = self._dashboard.take_snapshot()
        accuracy_before = self._curriculum.state.total_tasks_correct / max(
            1, self._curriculum.state.total_tasks_attempted
        )
        scores_before = self._expertise.export_scores()
        scores_before_float = {d: v["score"] for d, v in scores_before.items()}

        print(f"[Trainer] Starting training: {self._config.domains}")
        print(f"[Trainer] Tasks per domain: {self._config.tasks_per_domain}")

        for iteration in range(self._config.max_iterations):
            domain, level = self._curriculum.next_domain()
            print(f"[Trainer] Iteration {iteration+1}: domain={domain}, level={level}")

            tasks = self._task_gen.generate(
                domain=domain, difficulty=level, count=10
            )
            total_generated += len(tasks)

            for task in tasks:
                solve_result = self._solver.solve(task)
                total_solved += 1

                candidate = max(solve_result.candidates, key=lambda c: c.confidence)

                critique_result = self._critique.critique(task, candidate)

                correct, method = self._verifier.verify(task, candidate.answer)

                self._calibrator.record(
                    confidence=candidate.confidence,
                    correct=correct,
                    domain=domain,
                    difficulty=level,
                )
                self._curriculum.record_outcome(correct)

                if correct:
                    total_correct += 1

                experience = Experience(
                    experience_id=f"{task.task_id}-iter{iteration}",
                    domain=domain,
                    difficulty=level,
                    input=task.input,
                    candidate_reasoning=candidate.reasoning,
                    critique=critique_result.critique_reasoning,
                    correction="" if correct else "expected: " + task.expected_output,
                    final_answer=candidate.answer,
                    expected_answer=task.expected_output,
                    confidence=candidate.confidence,
                    verification_result=correct,
                    error_type="" if correct else "incorrect_answer",
                    training_status="verified" if correct else "error_analysis",
                    timestamp=time.time(),
                    model_version=self._version_manager.current_version,
                )

                self._experience.store(experience)

                if not correct:
                    hn_list = self._experience.generate_hard_negatives(experience, count=2)
                    hard_negatives += len(hn_list)
                    for hn in hn_list:
                        self._experience.store(hn)

            if self._experience.is_batch_ready():
                batch = self._experience.get_training_batch()
                print(f"[Trainer] Training batch ready: {len(batch)} verified experiences")
                self._experience.clear_batch()

            if (iteration + 1) % 20 == 0:
                self._curriculum.build_regression_suite(
                    self._generate_regression_tasks()
                )
                reg_result = self._curriculum.run_regression(
                    lambda t: self._solve_task(t)
                )
                if reg_result.get("regression_detected"):
                    print(f"[Trainer] REGRESSION DETECTED at iteration {iteration+1}!")
                else:
                    print(f"[Trainer] Regression OK: {reg_result.get('accuracy', 0):.1%}")

        snapshot_after = self._dashboard.take_snapshot()
        scores_after = self._expertise.export_scores()
        scores_after_float = {d: v["score"] for d, v in scores_after.items()}
        accuracy_after = total_correct / max(1, total_solved)

        self._curriculum.build_regression_suite(self._generate_regression_tasks())
        reg_final = self._curriculum.run_regression(lambda t: self._solve_task(t))

        version = self._version_manager.create_version(
            iteration=self._curriculum.state.iteration,
            training_tasks=total_generated,
            verified_experiences=total_correct,
            benchmark_accuracy=accuracy_after,
            domain_scores=scores_after_float,
            regression_passed=not reg_final.get("regression_detected", False),
            changes=f"Training run on {self._config.domains}",
        )

        self._dashboard.save_report(self._output_dir / "training_report.json")

        elapsed = time.perf_counter() - t0

        return TrainingResult(
            config=self._config,
            duration_seconds=elapsed,
            total_tasks_generated=total_generated,
            total_tasks_solved=total_solved,
            total_correct=total_correct,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            domain_scores_before=scores_before_float,
            domain_scores_after=scores_after_float,
            version=version.version_id,
            dashboard_report=self._dashboard.export_report(),
            regression_result=reg_final,
            calibration_summary=self._calibrator.calibration_summary(),
            experiences_stored=self._experience.get_stats()["total_stored"],
            hard_negatives_generated=hard_negatives,
        )

    def _generate_regression_tasks(self) -> list[dict[str, Any]]:
        tasks = []
        for domain in self._config.domains:
            gen_tasks = self._task_gen.generate(domain=domain, difficulty=2, count=5)
            for t in gen_tasks:
                tasks.append({
                    "task_id": t.task_id,
                    "input": t.input,
                    "expected_output": t.expected_output,
                    "domain": t.domain,
                })
        return tasks

    def _solve_task(self, task_dict: dict[str, Any]) -> str:
        task = Task(
            task_id=task_dict.get("task_id", "unknown"),
            domain=task_dict.get("domain", ""),
            difficulty=1,
            input=task_dict["input"],
            expected_output=task_dict["expected_output"],
            reasoning_type="deductive",
            generation_seed=0,
            verification_method="exact_value",
            metadata={},
        )
        result = self._solver.solve(task)
        return result.best_answer

    def run_quick(self, tasks_per_domain: int = 50) -> TrainingResult:
        """Run a quick training session for testing."""
        self._config.tasks_per_domain = tasks_per_domain
        self._config.max_iterations = 10
        self._config.regression_suite_size = 5
        return self.train()
