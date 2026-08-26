"""Deep debug of failing cases in basic_logic and noisy_input."""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import json
from benchmarks.dataset.generate import BenchmarkDataset
from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.neurons.centers import EvidenceGatherer, get_world_knowledge

ds = BenchmarkDataset(seed=42)
cases = ds.generate()
cortex = ReasoningCortex(enable_ml=False)
ev_gatherer = EvidenceGatherer()
wk = get_world_knowledge()

for cat in ("basic_logic", "noisy_input"):
    cat_cases = [c for c in cases if c.category == cat]
    fails = []
    for c in cat_cases:
        result = cortex.reason(query=c.query, evidence=c.evidence)
        if result.decision != c.expected_decision:
            fails.append((c, result))

    print(f"\n{'='*70}")
    print(f"{cat}: {len(cat_cases)-len(fails)}/{len(cat_cases)} correct ({len(fails)} fails)")
    print(f"{'='*70}")

    for c, r in fails[:15]:
        ev_text = "; ".join(c.evidence)[:100]
        print(f"\n--- {c.id}: {c.query}")
        print(f"  Evidence: {ev_text}")
        print(f"  Expected: {c.expected_decision} | Got: {r.decision} (conf={r.confidence:.3f})")

        # Trace direction detection
        for i, ev in enumerate(c.evidence):
            direction = ev_gatherer._detect_support_direction(ev, c.query)
            score = ev_gatherer._score_evidence(ev, {}, c.query)
            wk_check = wk.check_claim(ev)
            print(f"  Ev[{i}]: dir={direction} score={score:.3f} wk_plausible={wk_check.plausible} wk_conf={wk_check.confidence:.2f} wk_reason={wk_check.reasoning[:80]}")
