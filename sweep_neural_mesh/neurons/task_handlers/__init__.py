"""
Task Handlers — typed handlers for specific reasoning task categories.

Each handler is a focused module that processes a specific type of query
and returns a structured result.  The TaskRouter dispatches to the
appropriate handler based on query classification.

Categories:
  - logic:    deduction, induction, syllogisms, analogies, classification
  - math:     arithmetic, equations, word problems, verification
  - evidence: corroboration, contradiction, source ranking, entity resolution
  - temporal: date math, timeline, chronological ordering
  - causal:   chain reasoning, effect prediction, root cause analysis
"""
from .logic import LogicHandler, LogicResult
from .math import MathHandler, MathResult
from .evidence import EvidenceHandler, EvidenceResult
from .temporal import TemporalHandler, TemporalResult
from .causal import CausalHandler, CausalResult
from .router import TaskRouter, TaskClassification

__all__ = [
    "LogicHandler", "LogicResult",
    "MathHandler", "MathResult",
    "EvidenceHandler", "EvidenceResult",
    "TemporalHandler", "TemporalResult",
    "CausalHandler", "CausalResult",
    "TaskRouter", "TaskClassification",
]
