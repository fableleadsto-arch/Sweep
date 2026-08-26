"""
NeuralPacket — the universal internal representation of the Mesh.

All inter-node communication flows through NeuralPackets. They carry
tensors, embeddings, metadata, confidence, and provenance — decoupling
frameworks from one another.

    PyTorch Model
          ↓
    NeuralPacket
          ↓
    TensorFlow Model
          ↓
    NeuralPacket
          ↓
    ONNX Model
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .node import Modality


@dataclass
class Provenance:
    """Tracks the origin and transformation history of a packet."""
    source_node_id: str = ""
    source_node_name: str = ""
    source_model_version: str = ""
    parent_packet_id: str | None = None
    transformation: str = ""
    timestamp: float = field(default_factory=time.time)
    chain: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, node_id: str, node_name: str, transformation: str = "") -> Provenance:
        """Create a new provenance entry extending this chain."""
        return Provenance(
            source_node_id=node_id,
            source_node_name=node_name,
            parent_packet_id=None,
            transformation=transformation,
            chain=self.chain + [
                {
                    "node_id": self.source_node_id,
                    "node_name": self.source_node_name,
                    "transformation": self.transformation,
                    "timestamp": self.timestamp,
                }
            ],
        )


@dataclass
class SpatialContext:
    """Spatial metadata for image/video/audio packets."""
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    frame_index: int | None = None
    timestamp_ms: float | None = None
    duration_ms: float | None = None
    sample_rate: int | None = None


class NeuralPacket:
    """
    A framework-agnostic data packet for inter-node communication.

    NeuralPackets are the universal currency of the Mesh. They carry
    tensors (as raw lists or numpy-like arrays), embeddings, metadata,
    confidence scores, and provenance chains. Any model adapter can
    produce or consume them.
    """

    def __init__(
        self,
        data: Any = None,
        modality: Modality = Modality.TENSOR,
        packet_id: str | None = None,
        source_node_id: str = "",
        source_node_name: str = "",
        confidence: float = 0.0,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        spatial_context: SpatialContext | None = None,
        parent_packet_id: str | None = None,
    ):
        self.packet_id = packet_id or str(uuid.uuid4())[:16]
        self.modality = modality
        self.data = data
        self.embedding = embedding
        self.confidence = confidence
        self.metadata = metadata or {}
        self.spatial_context = spatial_context or SpatialContext()
        self.parent_packet_id = parent_packet_id
        self.created_at = time.time()
        self.provenance = Provenance(
            source_node_id=source_node_id,
            source_node_name=source_node_name,
            parent_packet_id=parent_packet_id,
        )

    # -- Convenience --

    def extend_provenance(
        self, node_id: str, node_name: str, transformation: str = ""
    ) -> NeuralPacket:
        """Return a copy with extended provenance (new packet_id)."""
        new = NeuralPacket(
            data=self.data,
            modality=self.modality,
            source_node_id=node_id,
            source_node_name=node_name,
            confidence=self.confidence,
            embedding=self.embedding,
            metadata=dict(self.metadata),
            spatial_context=self.spatial_context,
            parent_packet_id=self.packet_id,
        )
        new.provenance = self.provenance.extend(node_id, node_name, transformation)
        return new

    @property
    def provenance_chain(self) -> list[dict[str, Any]]:
        """Full chain of nodes that contributed to this packet."""
        return self.provenance.chain + [
            {
                "node_id": self.provenance.source_node_id,
                "node_name": self.provenance.source_node_name,
                "transformation": self.provenance.transformation,
                "timestamp": self.provenance.timestamp,
            }
        ]

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000

    # -- Serialisation --

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "modality": self.modality.value,
            "confidence": self.confidence,
            "has_data": self.data is not None,
            "has_embedding": self.embedding is not None,
            "embedding_dim": len(self.embedding) if self.embedding else 0,
            "metadata_keys": list(self.metadata.keys()),
            "parent_packet_id": self.parent_packet_id,
            "provenance_depth": len(self.provenance.chain),
            "age_ms": self.age_ms,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralPacket(id={self.packet_id}, modality={self.modality.value}, "
            f"conf={self.confidence:.3f}, src={self.provenance.source_node_name})"
        )
