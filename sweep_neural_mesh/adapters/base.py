"""
Base adapter — the interface all framework adapters implement.

An adapter wraps a native framework model and translates between
framework-specific tensors and NeuralPackets.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.node import Framework, Modality, NeuralNode, NodeSchema, NodeVersion
from ..core.packet import NeuralPacket


class BaseAdapter(ABC):
    """
    Abstract base for framework adapters.

    Each adapter:
    1. Loads a model from a path or object
    2. Executes inference
    3. Translates inputs/outputs to NeuralPacket form

    Adapters are framework-specific but the Mesh never sees the
    framework — only NeuralPackets.
    """

    framework: Framework = Framework.CUSTOM

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> Any:
        """Load a model and return a handle to it."""
        ...

    @abstractmethod
    def predict(self, model: Any, packet: NeuralPacket) -> NeuralPacket:
        """Run inference on a NeuralPacket and return the result."""
        ...

    @abstractmethod
    def validate_input(self, packet: NeuralPacket) -> bool:
        """Check whether this adapter can handle the given packet."""
        ...

    def wrap_as_node(
        self,
        model: Any,
        name: str,
        capabilities: list[str],
        schema: NodeSchema | None = None,
        version: NodeVersion | None = None,
    ) -> NeuralNode:
        """Create a NeuralNode that delegates to this adapter."""
        adapter = self

        def execute_fn(data: Any, **kwargs: Any) -> Any:
            packet = NeuralPacket(data=data, modality=(schema.output_modalities[0] if schema and schema.output_modalities else Modality.TENSOR))
            result = adapter.predict(model, packet)
            return result.data

        return NeuralNode(
            name=name,
            framework=self.framework,
            execute_fn=execute_fn,
            capabilities=capabilities,
            schema=schema or NodeSchema(),
            version=version or NodeVersion(),
        )
