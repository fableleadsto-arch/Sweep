"""
Self-Evolution — enables the neural mesh to learn and adapt.

Modules:
  - LearningModule:      tracks successes/failures, learns from feedback
  - EvolutionEngine:     mutates and evolves patterns
  - KnowledgeAcquisition: acquires new knowledge from external sources
  - PerformanceTracker:  monitors metrics and calibrates confidence
  - SelfEvolutionCoordinator: orchestrates all the above
"""
from .learning import LearningModule, LearningEvent
from .engine import EvolutionEngine, PatternMutation
from .knowledge import KnowledgeAcquisition
from .tracker import PerformanceTracker
from .coordinator import SelfEvolutionCoordinator

__all__ = [
    "LearningModule",
    "LearningEvent",
    "EvolutionEngine",
    "PatternMutation",
    "KnowledgeAcquisition",
    "PerformanceTracker",
    "SelfEvolutionCoordinator",
]
