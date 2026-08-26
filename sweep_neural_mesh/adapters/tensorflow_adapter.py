"""
TensorFlow adapter — wraps TensorFlow/Keras models for the Mesh.

Lazily imports tensorflow.
"""
from __future__ import annotations

from typing import Any

from ..core.node import Framework, NeuralNode
from ..core.packet import NeuralPacket
from .base import BaseAdapter


class TensorFlowAdapter(BaseAdapter):
    """Adapter for TensorFlow/Keras models."""

    framework = Framework.TENSORFLOW

    def load(self, path: str, **kwargs: Any) -> Any:
        import tensorflow as tf
        return tf.keras.models.load_model(path, **kwargs)

    def predict(self, model: Any, packet: NeuralPacket) -> NeuralPacket:
        import numpy as np
        input_data = self._to_array(packet.data)
        output = model(input_data, training=False)
        result_data = self._from_array(output)
        return NeuralPacket(
            data=result_data,
            modality=packet.modality,
            source_node_name=f"tensorflow:{type(model).__name__}",
            confidence=packet.confidence,
            metadata={**packet.metadata, "framework": "tensorflow"},
        )

    def validate_input(self, packet: NeuralPacket) -> bool:
        return packet.data is not None

    def _to_array(self, data: Any) -> Any:
        import numpy as np
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

    def _from_array(self, arr: Any) -> Any:
        if hasattr(arr, "numpy"):
            return arr.numpy().tolist()
        if hasattr(arr, "tolist"):
            return arr.tolist()
        return arr
