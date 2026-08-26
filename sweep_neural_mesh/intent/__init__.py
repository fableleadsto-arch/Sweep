"""
IntentMesh — bridges Sweep's intent system to the Neural Mesh.

Takes a classified intent and routes it through the appropriate
Mesh pipeline. This is the integration point between Sweep's
reasoning layer and its computation layer.

    User query
        ↓
    Intent Classifier (ml-service)
        ↓
    IntentMesh.route(intent, query)
        ↓
    Neural Mesh pipeline
        ↓
    Structured result with evidence
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..mesh import MeshConstraints, MeshResult, NeuralMesh
from ..core.node import Modality, NeuralNode, NodeCostProfile, NodeSchema
from ..core.packet import NeuralPacket
from ..registry.capability_registry import CapabilityRegistry

logger = logging.getLogger(__name__)


@dataclass
class IntentRoute:
    """Maps an intent to a Mesh capability pipeline."""
    intent: str
    capabilities: list[str]
    modalities: list[str]
    description: str = ""
    constraints: MeshConstraints | None = None


# Default routing table: intent → capability pipeline
DEFAULT_ROUTES: list[IntentRoute] = [
    IntentRoute(
        intent="lead_generation",
        capabilities=["data_extraction", "classification"],
        modalities=["text"],
        description="Extract and classify lead information",
    ),
    IntentRoute(
        intent="web_search",
        capabilities=["web_search", "summarization"],
        modalities=["text"],
        description="Search and summarize web content",
    ),
    IntentRoute(
        intent="web_analysis",
        capabilities=["web_analysis", "data_extraction"],
        modalities=["text"],
        description="Analyze web page structure and content",
    ),
    IntentRoute(
        intent="research",
        capabilities=["web_search", "summarization", "classification"],
        modalities=["text"],
        description="Deep research with multi-source synthesis",
    ),
    IntentRoute(
        intent="question_answering",
        capabilities=["question_answering"],
        modalities=["text"],
        description="Answer questions from knowledge or context",
    ),
    IntentRoute(
        intent="summarization",
        capabilities=["summarization"],
        modalities=["text"],
        description="Condense content into key points",
    ),
    IntentRoute(
        intent="code_generation",
        capabilities=["code_generation"],
        modalities=["text"],
        description="Generate code from natural language",
    ),
    IntentRoute(
        intent="code_analysis",
        capabilities=["code_analysis"],
        modalities=["text"],
        description="Analyze code for quality and issues",
    ),
    IntentRoute(
        intent="data_extraction",
        capabilities=["data_extraction"],
        modalities=["text"],
        description="Extract structured data from content",
    ),
    IntentRoute(
        intent="recommendation",
        capabilities=["recommendation", "classification"],
        modalities=["text"],
        description="Recommend tools, approaches, or solutions",
    ),
    IntentRoute(
        intent="classification",
        capabilities=["classification"],
        modalities=["text"],
        description="Classify content by type, topic, or priority",
    ),
    IntentRoute(
        intent="navigation",
        capabilities=["navigation"],
        modalities=["text"],
        description="Navigate to a specific page or section",
    ),
    IntentRoute(
        intent="voice_command",
        capabilities=["speech_to_text", "voice_command"],
        modalities=["audio", "text"],
        description="Process voice input and execute command",
    ),
    IntentRoute(
        intent="automation",
        capabilities=["automation", "workflow_creation"],
        modalities=["text"],
        description="Set up automated tasks or workflows",
    ),
    IntentRoute(
        intent="workflow_creation",
        capabilities=["workflow_creation"],
        modalities=["text"],
        description="Design new workflow pipelines",
    ),
    IntentRoute(
        intent="workflow_modification",
        capabilities=["workflow_modification"],
        modalities=["text"],
        description="Modify existing workflow configurations",
    ),
    IntentRoute(
        intent="analytics",
        capabilities=["analytics", "data_extraction"],
        modalities=["text"],
        description="Generate analytics reports and insights",
    ),
    IntentRoute(
        intent="general_chat",
        capabilities=["general_chat"],
        modalities=["text"],
        description="General conversation and assistance",
    ),
]


class IntentMesh:
    """
    Bridges intent classification to Neural Mesh execution.

    Maintains a routing table that maps intents to capability
    pipelines. When an intent is received, IntentMesh:
    1. Looks up the capability pipeline
    2. Checks which capabilities the Mesh can satisfy
    3. Constructs and executes the appropriate graph
    4. Returns a structured result
    """

    def __init__(self, mesh: NeuralMesh) -> None:
        self.mesh = mesh
        self._routes: dict[str, IntentRoute] = {}
        for route in DEFAULT_ROUTES:
            self._routes[route.intent] = route

    def register_route(self, route: IntentRoute) -> None:
        """Register or update a routing table entry."""
        self._routes[route.intent] = route

    def get_route(self, intent: str) -> IntentRoute | None:
        return self._routes.get(intent)

    def route(
        self,
        intent: str,
        query: str,
        context: dict[str, Any] | None = None,
        constraints: MeshConstraints | None = None,
    ) -> MeshResult:
        """
        Route an intent through the Mesh.

        Args:
            intent: Classified intent string.
            query: The original user query.
            context: Additional context (previous results, user preferences).
            constraints: Resource constraints.

        Returns:
            MeshResult with output, confidence, and evidence.
        """
        t0 = time.perf_counter()
        context = context or {}

        route = self._routes.get(intent)
        if route is None:
            # Unknown intent — try generic routing
            return self._route_generic(query, constraints)

        # Check which capabilities are available
        available_caps = []
        for cap in route.capabilities:
            nodes = self.mesh.registry.find_capability(cap)
            if nodes:
                available_caps.append(cap)

        if not available_caps:
            # No matching capabilities — fall back to general chat
            return self._route_generic(query, constraints)

        # Execute through mesh with first available capability
        # (full pipeline execution would chain capabilities)
        primary_cap = available_caps[0]
        result = self.mesh.analyze(
            data=query,
            task=primary_cap,
            modalities=route.modalities,
            constraints=constraints or route.constraints,
        )

        # Enrich with intent metadata
        result.metadata = {
            "intent": intent,
            "route_capabilities": route.capabilities,
            "available_capabilities": available_caps,
            "description": route.description,
        }
        result.evidence.append({
            "type": "intent_classification",
            "intent": intent,
            "confidence": result.confidence,
        })

        return result

    def _route_generic(
        self,
        query: str,
        constraints: MeshConstraints | None,
    ) -> MeshResult:
        """Fallback routing for unknown intents."""
        return self.mesh.analyze(
            data=query,
            task="general_chat",
            modalities=["text"],
            constraints=constraints,
        )

    def available_intents(self) -> list[str]:
        return sorted(self._routes.keys())

    def satisfied_intents(self) -> list[str]:
        """Intents whose capabilities are all available in the Mesh."""
        satisfied = []
        for intent, route in self._routes.items():
            all_available = all(
                bool(self.mesh.registry.find_capability(cap))
                for cap in route.capabilities
            )
            if all_available:
                satisfied.append(intent)
        return satisfied

    def coverage_report(self) -> dict[str, Any]:
        """Report which intents can be fully served by the Mesh."""
        total = len(self._routes)
        satisfied = len(self.satisfied_intents())
        return {
            "total_intents": total,
            "satisfied": satisfied,
            "coverage": satisfied / total if total > 0 else 0,
            "unsatisfied": [
                intent for intent in self._routes
                if intent not in self.satisfied_intents()
            ],
        }

    def __repr__(self) -> str:
        return f"IntentMesh(routes={len(self._routes)})"
