"""
PyTorch adapter — wraps PyTorch models for the Mesh.

Lazily imports torch. If torch is not installed, the adapter
is still importable but raises ImportError on use.
"""
from __future__ import annotations

from typing import Any

from ..core.node import Framework, Modality, NeuralNode, NodeSchema, NodeVersion
from ..core.packet import NeuralPacket
from .base import BaseAdapter


class PyTorchAdapter(BaseAdapter):
    """Adapter for PyTorch models."""

    framework = Framework.PYTORCH

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def load(self, path: str, **kwargs: Any) -> Any:
        import torch
        model = torch.load(path, map_location=self.device, **kwargs)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def predict(self, model: Any, packet: NeuralPacket) -> NeuralPacket:
        import torch
        with torch.no_grad():
            input_data = self._to_tensor(packet.data)
            output = model(input_data)
            result_data = self._from_tensor(output)
        return NeuralPacket(
            data=result_data,
            modality=packet.modality,
            source_node_name=f"pytorch:{type(model).__name__}",
            confidence=packet.confidence,
            metadata={**packet.metadata, "framework": "pytorch"},
        )

    def validate_input(self, packet: NeuralPacket) -> bool:
        return packet.data is not None

    def _to_tensor(self, data: Any) -> Any:
        import torch
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, list):
            return torch.tensor(data, device=self.device)
        if isinstance(data, (int, float)):
            return torch.tensor([data], device=self.device)
        return data

    def _from_tensor(self, tensor: Any) -> Any:
        if hasattr(tensor, "cpu"):
            return tensor.cpu().tolist()
        return tensor
