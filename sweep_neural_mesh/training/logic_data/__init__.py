"""Logic Data pipeline: real open datasets -> Sweep Tasks -> RSI.

Exposes the loader (fetch+convert) and knowledge mode (chunk-feed into Sweep).
"""
from sweep_neural_mesh.training.logic_data.loader import (
    build_dataset,
    proofwriter_rows,
    logiqa_rows,
    ruletaker_rows,
    folio_rows,
)
from sweep_neural_mesh.training.logic_data.knowledge import KnowledgeFeed

__all__ = [
    "build_dataset",
    "proofwriter_rows",
    "logiqa_rows",
    "ruletaker_rows",
    "folio_rows",
    "KnowledgeFeed",
]
