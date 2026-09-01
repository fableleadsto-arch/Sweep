"""
Self-Evolution Module — compatibility shim.

The implementation has been moved to ``neurons/evolution/``.
This module re-exports all public names so existing imports keep working.
"""
from .evolution import (  # noqa: F401
    LearningEvent,
    LearningModule,
    EvolutionEngine,
    PatternMutation,
    KnowledgeAcquisition,
    PerformanceTracker,
    SelfEvolutionCoordinator,
)

__all__ = [
    "LearningEvent",
    "LearningModule",
    "EvolutionEngine",
    "PatternMutation",
    "KnowledgeAcquisition",
    "PerformanceTracker",
    "SelfEvolutionCoordinator",
]
