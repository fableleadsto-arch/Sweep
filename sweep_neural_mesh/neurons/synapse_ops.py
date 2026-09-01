"""
Synapse Operations — synaptic plasticity, Hebbian learning, myelination.

Extracted from cortex.py to reduce its size.
Manages synaptic connections between processing centers.
"""
from __future__ import annotations

from typing import Any

from .signal import Signal, Synapse, SynapseType
from .plasticity import SynapticPlasticity


def build_default_synapses() -> dict[str, Synapse]:
    """Build the default synaptic connections between centers."""
    synapses: dict[str, Synapse] = {}
    for fc, tc, wt, st in [
        ("evidence_gatherer", "credibility_assessor", 0.9, SynapseType.EXCITATORY),
        ("evidence_gatherer", "temporal_sequencer", 0.8, SynapseType.EXCITATORY),
        ("evidence_gatherer", "causal_linker", 0.85, SynapseType.EXCITATORY),
        ("evidence_gatherer", "contradiction_detector", 0.8, SynapseType.EXCITATORY),
        ("credibility_assessor", "causal_linker", 0.7, SynapseType.MODULATORY),
        ("contradiction_detector", "explanation_builder", 0.6, SynapseType.INHIBITORY),
    ]:
        synapses[f"{fc}->{tc}"] = Synapse(
            from_center=fc, to_center=tc, weight=wt, synapse_type=st)
    return synapses


def apply_synaptic_input(
    center_name: str,
    signals: list[Signal],
    synapses: dict[str, Synapse],
) -> list[Signal]:
    """Modulate signals by synaptic weights before entering a center."""
    modulated = []
    for sig in signals:
        for syn in synapses.values():
            if syn.to_center == center_name:
                sig = syn.transmit(sig)
        modulated.append(sig)
    return modulated


def hebbian_update(
    center_name: str,
    output: list[Signal],
    synapses: dict[str, Synapse],
) -> None:
    """Apply Hebbian learning: strengthen synapses with high-quality output."""
    if not output:
        return
    avg = sum(s.confidence for s in output) / len(output)
    for syn in synapses.values():
        if syn.to_center == center_name:
            if avg > 0.4:
                syn.strengthen(0.02)
            elif avg < 0.2:
                syn.weaken(0.01)


def update_synaptic_plasticity(
    final_conf: float,
    cd: dict[str, Any],
    synapses: dict[str, Synapse],
) -> None:
    """Update synapse weights based on overall reasoning quality."""
    quality = 0.5
    if final_conf > 0.7 and cd.get("decision") != "insufficient":
        quality = 0.8
    elif final_conf < 0.3:
        quality = 0.2
    for syn in synapses.values():
        if quality > 0.6:
            syn.strengthen(0.01 * quality)
        elif quality < 0.3:
            syn.weaken(0.01 * (1.0 - quality))


def apply_myelination(
    center_times: dict[str, float],
    plasticity: SynapticPlasticity,
) -> None:
    """Apply myelination speedup to fast-processing centers."""
    for name, elapsed in center_times.items():
        plasticity.apply_myelination_speedup(name, elapsed)
