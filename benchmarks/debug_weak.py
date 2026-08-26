from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import json
from benchmarks.dataset.generate import BenchmarkDataset
from benchmarks.runners.sweep_runner import SweepRunner

ds = BenchmarkDataset(seed=42)
cases = ds.generate()

for cat in ("basic_logic", "noisy_input"):
    cat_cases = [c for c in cases if c.category == cat]
    runner = SweepRunner(enable_ml=False)
    data = runner.run_all(cat_cases, verbose=False)
    fails = [r for r in data["results"] if not r["decision_correct"]]
    print(f"\n{'='*60}")
    print(f"{cat}: {data['summary']['accuracy']:.1%} ({len(cat_cases)-len(fails)}/{len(cat_cases)})")
    print(f"{'='*60}")
    for f in fails[:30]:
        print(f"\nID: {f['id']}")
        print(f"  Q: {f['query']}")
        print(f"  Evidence: {f['expected_answer'][:80]}...")
        print(f"  Expected: {f['expected_decision']}  Actual: {f['actual_decision']}  Conf: {f['actual_confidence']:.3f}")
        print(f"  Reasoning: {f['actual_reasoning'][:120]}")
