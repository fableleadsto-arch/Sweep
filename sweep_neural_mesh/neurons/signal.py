"""
Signal and Synapse — the fundamental units of neuronal communication.

A Signal is the computational equivalent of a neurotransmitter packet:
it carries data, confidence, and metadata through the neural network.

A Synapse is the connection between processing centers:
it has a weight (strength), a type (excitatory/inhibitory),
and plasticity (how it changes with use).

The flow:

    Input Signal
        ↓  [Synapse: weight × signal]
    Processing Center
        ↓  [Synapse: weight × signal]
    Output Signal
        ↓
    Next Center
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("sweep.neurons.signal")


class SignalType(Enum):
    """What kind of signal this is — determines how centers process it."""
    EVIDENCE = "evidence"
    CREDIBILITY = "credibility"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CONTRADICTION = "contradiction"
    INTEGRATED = "integrated"
    CONSENSUS = "consensus"
    EXPLANATION = "explanation"
    RAW = "raw"


class SynapseType(Enum):
    """How the synapse modifies the signal passing through it."""
    EXCITATORY = "excitatory"   # strengthens the signal
    INHIBITORY = "inhibitory"   # weakens or suppresses the signal
    MODULATORY = "modulatory"   # changes signal characteristics (not just amplitude)


@dataclass
class Signal:
    """
    A unit of information flowing through the neural network.

    Analogous to a neurotransmitter packet crossing a synapse:
    it carries payload data, a confidence level, and metadata
    about its origin and journey through the network.
    """
    data: dict[str, any]
    signal_type: SignalType = SignalType.RAW
    confidence: float = 1.0          # 0.0–1.0: how sure we are about this signal
    source_center: str = ""          # which center produced this
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)  # centers this signal passed through
    # ── New fields for embedding + urgency ──
    embedding_bits: int = 0          # SimHash fingerprint for semantic similarity
    urgency: float = 0.5             # 0.0–1.0: how urgent is this signal
    ttl: float = 300.0               # time-to-live in seconds before decay
    emotional_valence: float = 0.0   # -1.0 (threat) to 1.0 (reward)

    def amplify(self, factor: float) -> Signal:
        """Strengthen this signal (excitatory synapse effect)."""
        new_confidence = min(1.0, self.confidence * factor)
        logger.debug(f"Signal {self.signal_id} amplified by {factor:.2f}: {self.confidence:.3f} → {new_confidence:.3f}")
        return Signal(
            data=self.data,
            signal_type=self.signal_type,
            confidence=new_confidence,
            source_center=self.source_center,
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            metadata={**self.metadata, "amplified_by": factor},
            history=list(self.history),
            embedding_bits=self.embedding_bits,
            urgency=min(1.0, self.urgency * factor),
            ttl=self.ttl,
            emotional_valence=self.emotional_valence,
        )

    def dampen(self, factor: float) -> Signal:
        """Weaken this signal (inhibitory synapse effect)."""
        new_confidence = max(0.0, self.confidence * factor)
        logger.debug(f"Signal {self.signal_id} dampened by {factor:.2f}: {self.confidence:.3f} → {new_confidence:.3f}")
        return Signal(
            data=self.data,
            signal_type=self.signal_type,
            confidence=new_confidence,
            source_center=self.source_center,
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            metadata={**self.metadata, "dampened_by": factor},
            history=list(self.history),
            embedding_bits=self.embedding_bits,
            urgency=self.urgency,
            ttl=self.ttl,
            emotional_valence=self.emotional_valence,
        )

    def stamp(self, center_name: str) -> Signal:
        """Record that this signal passed through a center."""
        return Signal(
            data=self.data,
            signal_type=self.signal_type,
            confidence=self.confidence,
            source_center=self.source_center or center_name,
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
            history=self.history + [center_name],
            embedding_bits=self.embedding_bits,
            urgency=self.urgency,
            ttl=self.ttl,
            emotional_valence=self.emotional_valence,
        )

    def to_dict(self) -> dict[str, any]:
        return {
            "signal_id": self.signal_id,
            "type": self.signal_type.value,
            "confidence": self.confidence,
            "source_center": self.source_center,
            "data_keys": list(self.data.keys()) if isinstance(self.data, dict) else [],
            "history": self.history,
            "urgency": round(self.urgency, 3),
            "has_embedding": self.embedding_bits != 0,
        }


@dataclass
class Synapse:
    """
    A weighted connection between two processing centers.

    Analogous to a biological synapse: it has a strength (weight),
    a type (excitatory/inhibitory), and plasticity (adapts with use).
    """
    from_center: str
    to_center: str
    weight: float = 1.0             # 0.0–2.0: signal multiplier
    synapse_type: SynapseType = SynapseType.EXCITATORY
    plasticity: float = 0.1         # how much weight changes per activation
    activation_count: int = 0
    total_weight_change: float = 0.0

    def transmit(self, signal: Signal) -> Signal:
        """Pass a signal through this synapse, applying weight and type effects."""
        self.activation_count += 1
        logger.debug(f"Synapse {self.from_center}→{self.to_center} transmitting {signal.signal_id} "
                     f"(type={self.synapse_type.value}, w={self.weight:.3f}, activations={self.activation_count})")

        if self.synapse_type == SynapseType.EXCITATORY:
            # Excitatory: fill in the "missing confidence" proportionally
            # weight=1.0 fills 30% of the gap, weight=0.0 fills nothing
            boost = (1.0 - signal.confidence) * self.weight * 0.3
            new_conf = min(1.0, signal.confidence + boost)
            modified = Signal(
                data=signal.data,
                signal_type=signal.signal_type,
                confidence=new_conf,
                source_center=signal.source_center,
                signal_id=signal.signal_id,
                timestamp=signal.timestamp,
                metadata={**signal.metadata, "excitatory_boost": boost},
                history=list(signal.history),
            )
        elif self.synapse_type == SynapseType.INHIBITORY:
            # Inhibitory: reduce confidence proportionally
            reduction = signal.confidence * self.weight * 0.3
            new_conf = max(0.0, signal.confidence - reduction)
            modified = Signal(
                data=signal.data,
                signal_type=signal.signal_type,
                confidence=new_conf,
                source_center=signal.source_center,
                signal_id=signal.signal_id,
                timestamp=signal.timestamp,
                metadata={**signal.metadata, "inhibitory_reduction": reduction},
                history=list(signal.history),
            )
        else:
            # Modulatory: weight changes the signal's metadata, not confidence
            modified = Signal(
                data=signal.data,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                source_center=signal.source_center,
                signal_id=signal.signal_id,
                timestamp=signal.timestamp,
                metadata={**signal.metadata, "modulated_by": self.from_center, "modulation_weight": self.weight},
                history=list(signal.history),
            )

        return modified.stamp(f"synapse:{self.from_center}->{self.to_center}")

    def strengthen(self, amount: float | None = None) -> None:
        """Hebbian learning: strengthen this synapse (use strengthens connection)."""
        delta = amount if amount is not None else self.plasticity
        self.weight = min(2.0, self.weight + delta)
        self.total_weight_change += delta

    def weaken(self, amount: float | None = None) -> None:
        """Weaken this synapse (unused connections decay)."""
        delta = amount if amount is not None else self.plasticity
        self.weight = max(0.0, self.weight - delta)
        self.total_weight_change -= delta

    def to_dict(self) -> dict[str, any]:
        return {
            "from": self.from_center,
            "to": self.to_center,
            "weight": round(self.weight, 4),
            "type": self.synapse_type.value,
            "activations": self.activation_count,
            "plasticity": self.plasticity,
        }
