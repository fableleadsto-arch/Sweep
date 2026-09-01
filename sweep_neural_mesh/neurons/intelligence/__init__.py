"""
Intelligence — gathering, organizing, and understanding information.

Modules:
  - Gatherer:    collect information from multiple sources
  - Organizer:   structure, categorize, and deduplicate
  - Analyzer:    reason about and extract insights
  - Store:       persist and retrieve organized intelligence
  - Pipeline:    orchestrate the full flow
"""
from .gatherer import IntelligenceGatherer, GatheredIntel
from .organizer import IntelligenceOrganizer, OrganizedIntel
from .analyzer import IntelligenceAnalyzer, AnalyzedIntel
from .store import IntelligenceStore
from .pipeline import IntelligencePipeline

__all__ = [
    "IntelligenceGatherer",
    "GatheredIntel",
    "IntelligenceOrganizer",
    "OrganizedIntel",
    "IntelligenceAnalyzer",
    "AnalyzedIntel",
    "IntelligenceStore",
    "IntelligencePipeline",
]
