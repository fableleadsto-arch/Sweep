"""
NeuralMesh — the top-level API for the Sweep Neural Mesh.

The Mesh is the computational fabric between Sweep's intent layer and
its specialized ML models. It provides:

    mesh.analyze(data, task, constraints) -> MeshResult

The rest of Sweep should never need to know whether the result came
from PyTorch, TensorFlow, ONNX, scikit-learn, CPU, GPU, one model,
or five models.

Usage:

    from sweep_neural_mesh import NeuralMesh

    mesh = NeuralMesh()
    mesh.register_node(vision_node)
    mesh.register_node(audio_node)

    result = mesh.analyze(
        data=image_bytes,
        task="face_detection",
        modalities=["image"],
    )
    print(result.confidence, result.output)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .adapters.base import BaseAdapter
from .core.engine import ExecutionEngine, ExecutionResult
from .core.graph import MeshGraph
from .core.node import (
    Framework,
    Modality,
    NeuralNode,
    NodeCostProfile,
    NodeResult,
    NodeSchema,
    NodeVersion,
)
from .core.packet import NeuralPacket
from .core.router import ModelRouter, RoutingContext, RoutingResult
from .fusion import FusionEngine
from .fusion.confidence import ConfidenceEngine, ConfidenceReport
from .fusion.verification import VerificationEngine, VerificationResult
from .memory import FeatureCache
from .registry import ModelRegistry
from .registry.capability_registry import CapabilityRegistry
from .resources import ResourceManager
from .telemetry import Telemetry

logger = logging.getLogger(__name__)


@dataclass
class MeshConstraints:
    """Resource and quality constraints for an analysis request."""
    latency_budget_ms: float = 5000.0
    memory_budget_mb: float = 4096.0
    require_gpu: bool = False
    preferred_frameworks: list[Framework] = field(default_factory=list)
    quality_threshold: float = 0.5
    fusion_strategy: str = "confidence_weighted"
    verification_enabled: bool = False


@dataclass
class MeshResult:
    """The result of a mesh.analyze() call."""
    success: bool = True
    output: Any = None
    confidence: float = 0.0
    confidence_report: ConfidenceReport | None = None
    verification: VerificationResult | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    nodes_used: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    resource_usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    raw_packets: list[NeuralPacket] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "confidence": self.confidence,
            "nodes_used": self.nodes_used,
            "models_used": self.models_used,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
            "output_type": type(self.output).__name__,
            "evidence_count": len(self.evidence),
        }


class NeuralMesh:
    """
    The top-level Neural Mesh API.

    Orchestrates node registration, capability discovery, routing,
    execution, fusion, verification, and telemetry. This is the only
    class the rest of Sweep needs to import.
    """

    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.capabilities = CapabilityRegistry()
        self.router = ModelRouter(self.registry)
        self.engine = ExecutionEngine()
        self.fusion = FusionEngine()
        self.confidence = ConfidenceEngine()
        self.verification = VerificationEngine()
        self.resources = ResourceManager()
        self.telemetry = Telemetry()
        self.cache = FeatureCache()

    # -- Registration --

    def register_node(self, node: NeuralNode) -> None:
        """Register a node with the Mesh."""
        self.registry.register(node)
        self.telemetry.inc_counter("nodes_registered")
        logger.info("Registered node: %s (%s)", node.name, node.node_id)

    def register_nodes(self, nodes: list[NeuralNode]) -> None:
        for n in nodes:
            self.register_node(n)

    def register_adapter(
        self,
        adapter: BaseAdapter,
        model: Any,
        name: str,
        capabilities: list[str],
        schema: NodeSchema | None = None,
    ) -> NeuralNode:
        """Register a model via an adapter — wraps it as a NeuralNode."""
        node = adapter.wrap_as_node(model, name, capabilities, schema)
        self.register_node(node)
        return node

    def register_capability(
        self,
        name: str,
        modality: Modality,
        description: str = "",
        prerequisites: list[str] | None = None,
    ) -> None:
        self.capabilities.register(name, modality, description, prerequisites)

    # -- Analysis --

    def analyze(
        self,
        data: Any,
        task: str,
        modalities: list[str] | None = None,
        constraints: MeshConstraints | None = None,
    ) -> MeshResult:
        """
        The primary Mesh API.

        Analyzes input data for a given task, using the registered
        nodes, capabilities, and constraints.

        Returns a MeshResult with output, confidence, provenance,
        and telemetry.
        """
        t0 = time.perf_counter()
        constraints = constraints or MeshConstraints()
        result = MeshResult()
        modality_enums = [Modality(m) for m in (modalities or ["tensor"])]

        # Check cache
        cache_key = f"{task}:{str(hash(str(data)))[:16]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.telemetry.inc_counter("cache_hits")
            result.output = cached
            result.confidence = 1.0
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.warnings.append("served from cache")
            return result

        # Find capable nodes
        context = RoutingContext(
            task=task,
            required_capability=task,
            latency_budget_ms=constraints.latency_budget_ms,
            memory_budget_mb=constraints.memory_budget_mb,
            require_gpu=constraints.require_gpu,
            preferred_frameworks=constraints.preferred_frameworks,
        )
        routing = self.router.rank_multi(task, context, top_k=3)

        if routing.selected_node is None:
            result.success = False
            result.warnings.append(f"no node found for capability: {task}")
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # Execute
        packet = NeuralPacket(
            data=data,
            modality=modality_enums[0] if modality_enums else Modality.TENSOR,
            source_node_name="mesh_input",
        )

        self.telemetry.record("routing", node_name=routing.selected_node.name)

        exec_result = self.engine.execute(
            self._build_single_node_graph(routing.selected_node),
            initial_inputs={routing.selected_node.node_id: packet},
        )

        # Collect results
        all_nodes = [routing.selected_node] + routing.alternatives
        node_confidences: dict[str, float] = {}
        packets: list[NeuralPacket] = []

        if exec_result.output_packets:
            packets = exec_result.output_packets
            for p in packets:
                node_confidences[p.provenance.source_node_name] = p.confidence
        elif exec_result.node_results.get(routing.selected_node.node_id):
            nr = exec_result.node_results[routing.selected_node.node_id]
            node_confidences[routing.selected_node.name] = nr.confidence
            out_packet = NeuralPacket(
                data=nr.output,
                confidence=nr.confidence,
                source_node_id=routing.selected_node.node_id,
                source_node_name=routing.selected_node.name,
            )
            packets.append(out_packet)

        # Confidence
        conf_report = self.confidence.evaluate(node_confidences)
        result.confidence_report = conf_report
        result.confidence = conf_report.score

        # Output
        if packets:
            primary = packets[0]
            result.output = primary.data
            result.raw_packets = packets
            result.provenance = primary.provenance_chain

        # Verification
        if constraints.verification_enabled and len(packets) > 1:
            outputs = [p.data for p in packets]
            sources = [p.provenance.source_node_name for p in packets]
            result.verification = self.verification.verify(outputs, sources)

        # Evidence
        for p in packets:
            result.evidence.append({
                "source": p.provenance.source_node_name,
                "confidence": p.confidence,
                "modality": p.modality.value,
                "provenance_depth": len(p.provenance.chain),
            })

        # Nodes/models used
        result.nodes_used = list(set(
            p.provenance.source_node_id for p in packets if p.provenance.source_node_id
        ))
        result.models_used = list(set(
            p.provenance.source_node_name for p in packets
        ))

        # Cache result
        self.cache.put(cache_key, result.output)

        # Telemetry
        result.latency_ms = (time.perf_counter() - t0) * 1000
        self.telemetry.record(
            "analysis",
            node_name=task,
            duration_ms=result.latency_ms,
            success=result.success,
            confidence=result.confidence,
        )

        # Resource snapshot
        result.resource_usage = self.resources.profile.to_dict()

        return result

    # -- Helpers --

    def _build_single_node_graph(self, node: NeuralNode) -> MeshGraph:
        graph = MeshGraph(name=f"single:{node.name}")
        graph.add_node(node)
        return graph

    # -- Stats --

    def summary(self) -> dict[str, Any]:
        return {
            "registry": self.registry.summary(),
            "capabilities": self.capabilities.summary(),
            "engine": self.engine.stats,
            "telemetry": self.telemetry.summary(),
            "resources": self.resources.profile.to_dict(),
            "workload_state": self.resources.state.value,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralMesh(nodes={self.registry.size}, "
            f"caps={len(self.capabilities.all_names)}, "
            f"state={self.resources.state.value})"
        )
