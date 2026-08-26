"""
Scikit-learn adapter — wraps sklearn models for the Mesh.

This adapter handles the ML service's intent classifier and
any other scikit-learn pipelines.
"""
from __future__ import annotations

from typing import Any

from ..core.node import Framework
from ..core.packet import NeuralPacket
from .base import BaseAdapter


class SklearnAdapter(BaseAdapter):
    """Adapter for scikit-learn models."""

    framework = Framework.SKLEARN

    def load(self, path: str, **kwargs: Any) -> Any:
        import joblib
        return joblib.load(path)

    def predict(self, model: Any, packet: NeuralPacket) -> NeuralPacket:
        input_data = packet.data
        if isinstance(input_data, str):
            input_data = [input_data]
        elif not isinstance(input_data, list):
            input_data = [input_data]

        output = model.predict(input_data)
        result = output.tolist() if hasattr(output, "tolist") else list(output)

        # Get probabilities if available
        confidence = packet.confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            max_proba = float(max(max(p) for p in proba))
            confidence = max_proba

        return NeuralPacket(
            data=result,
            modality=packet.modality,
            source_node_name="sklearn:Pipeline",
            confidence=confidence,
            metadata={**packet.metadata, "framework": "sklearn"},
        )

    def validate_input(self, packet: NeuralPacket) -> bool:
        return packet.data is not None
