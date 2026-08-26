"""
First Training Experiment — §34: Run training on 3 weak domains.

Target domains: basic_logic, novel_structures, ambiguity
1000 tasks per domain, verified task generation, error-driven learning.
"""
from __future__ import annotations

import sys
import io
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from sweep_neural_mesh.training.trainer import Trainer, TrainingConfig


def run_experiment() -> None:
    print("=" * 60)
    print("  SWEEP FIRST TRAINING EXPERIMENT")
    print("  §34: 1000 tasks × 3 weak domains")
    print("=" * 60)
    print()

    config = TrainingConfig(
        domains=["basic_logic", "novel_structures", "ambiguity"],
        tasks_per_domain=1000,
        difficulty_level=2,
        num_candidates=3,
        regression_suite_size=20,
        batch_size=100,
        max_iterations=50,
        mastery_threshold=0.90,
        output_dir="sweep_neural_mesh/training/results/experiment_1",
    )

    trainer = Trainer(config)

    print("[Experiment] Running training...")
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("  EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"  Duration:               {elapsed:.1f}s")
    print(f"  Tasks generated:        {result.total_tasks_generated}")
    print(f"  Tasks solved:           {result.total_tasks_solved}")
    print(f"  Correct:                {result.total_correct}")
    print(f"  Accuracy before:        {result.accuracy_before:.1%}")
    print(f"  Accuracy after:         {result.accuracy_after:.1%}")
    print(f"  Version:                {result.version}")
    print(f"  Experiences stored:     {result.experiences_stored}")
    print(f"  Hard negatives:         {result.hard_negatives_generated}")
    print()
    print("DOMAIN SCORES (before → after):")
    all_domains = set(list(result.domain_scores_before.keys()) +
                      list(result.domain_scores_after.keys()))
    for domain in sorted(all_domains):
        before = result.domain_scores_before.get(domain, 0.0)
        after = result.domain_scores_after.get(domain, 0.0)
        delta = after - before
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {domain:<28} {before:.1%} → {after:.1%} {arrow}{abs(delta):.1%}")
    print()
    print("REGRESSION:")
    reg = result.regression_result
    print(f"  Passed:   {not reg.get('regression_detected', True)}")
    print(f"  Accuracy: {reg.get('accuracy', 0):.1%}")
    print()
    print("CALIBRATION:")
    cal = result.calibration_summary
    print(f"  ECE:      {cal.get('ece', 0):.4f}")
    print(f"  Overconf: {cal.get('overconfidence_penalty', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
