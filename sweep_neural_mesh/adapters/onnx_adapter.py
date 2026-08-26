"""
ONNX Runtime adapter — wraps ONNX models for the Mesh.

Lazily imports onnxruntime.
"""
from __future__ import annotations

from typing import Any

from ..core.node import Framework
from ..core.packet import NeuralPacket
from .base import BaseAdapter


class ONNXAdapter(BaseAdapter):
    """Adapter for ONNX Runtime models."""

    framework = Framework.ONNX

    def __init__(self, providers: list[str] | None = None) -> None:
        self.providers = providers or ["CPUExecutionProvider"]

    def load(self, path: str, **kwargs: Any) -> Any:
        import onnxruntime as ort
        session_options = kwargs.pop("session_options", None)
        if session_options:
            return ort.InferenceSession(path, sess_options=session_options, providers=self.providers)
        return ort.InferenceSession(path, providers=self.providers)

    def predict(self, model: Any, packet: NeuralPacket) -> NeuralPacket:
        import numpy as np
        input_data = self._to_array(packet.data)
        input_name = model.get_inputs()[0].name
        output = model.run(None, {input_name: input_data})
        result_data = [o.tolist() if hasattr(o, "tolist") else o for o in output]
        if len(result_data) == 1:
            result_data = result_data[0]
        return NeuralPacket(
            data=result_data,
            modality=packet.modality,
            source_node_name="onnx:InferenceSession",
            confidence=packet.confidence,
            metadata={**packet.metadata, "framework": "onnx"},
        )

    def validate_input(self, packet: NeuralPacket) -> bool:
        return packet.data is not None

    def _to_array(self, data: Any) -> Any:
        import numpy as np
        if isinstance(data, np.ndarray):
            return data
        return np.array(data, dtype=np.float32)
