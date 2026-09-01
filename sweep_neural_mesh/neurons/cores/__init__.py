"""
Neural Cores — specialized processing units that run in parallel.

Each core handles one aspect of reasoning:
  - FactualCore:   knowledge lookup (facts, numbers, names)
  - ReasoningCore: logic, common sense, yes/no, math
  - EvidenceCore:  evidence matching and extraction
  - TemporalCore:  dates, time, historical events
  - CausalCore:    cause-effect chains

All cores implement NeuralCoreProtocol from core_protocol.py.
"""
from .factual_core import FactualCore
from .reasoning_core import ReasoningCore
from .evidence_core import EvidenceCore
from .temporal_core import TemporalCore
from .causal_core import CausalCore
from .coordinator import MultiCoreCoordinator

__all__ = [
    "FactualCore",
    "ReasoningCore",
    "EvidenceCore",
    "TemporalCore",
    "CausalCore",
    "MultiCoreCoordinator",
]
