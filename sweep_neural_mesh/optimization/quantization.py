"""
Quantization — precision reduction for model optimization.

Reduces model size and improves inference speed by lowering numeric
precision (float64 → float32 → float16 → int8). This module operates
on packet-level data, not on raw model weights — it quantizes the
Mesh's internal NeuralPacket representations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class QuantizationProfile:
    """Result of quantizing a dataset or tensor."""
    original_dtype: str
    quantized_dtype: str
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    max_error: float
    mean_error: float
    samples_processed: int


class Quantizer:
    """
    Quantizes NeuralPacket data to lower precision.

    Supports:
    - float64 → float32 (halves memory)
    - float32 → float16 (quarters memory, some精度 loss)
    - float32 → int8 (8x compression, significant loss)
    - Dynamic range quantization (symmetric min/max)
    """

    def __init__(self) -> None:
        self._profiles: list[QuantizationProfile] = []

    def quantize(
        self,
        data: Any,
        target_precision: str = "float32",
    ) -> tuple[Any, QuantizationProfile]:
        """
        Quantize data to target precision.

        Returns (quantized_data, profile).
        """
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return self._quantize_vector(data, target_precision)
        if isinstance(data, list) and isinstance(data[0], list):
            return self._quantize_matrix(data, target_precision)
        return data, QuantizationProfile(
            original_dtype="unknown",
            quantized_dtype="unknown",
            original_size_bytes=0,
            quantized_size_bytes=0,
            compression_ratio=1.0,
            max_error=0.0,
            mean_error=0.0,
            samples_processed=0,
        )

    def _quantize_vector(
        self, vec: list[float], target: str
    ) -> tuple[list, QuantizationProfile]:
        original_bytes = len(vec) * 8  # float64 assumption

        if target == "float32":
            quantized = [float(x) for x in vec]
            quantized_bytes = len(vec) * 4
            errors = [abs(a - b) for a, b in zip(vec, quantized)]
        elif target == "float16":
            # Simulate float16 precision (3 decimal digits)
            quantized = [round(float(x), 3) for x in vec]
            quantized_bytes = len(vec) * 2
            errors = [abs(a - b) for a, b in zip(vec, quantized)]
        elif target == "int8":
            # Symmetric min/max quantization
            min_val = min(vec) if vec else 0
            max_val = max(vec) if vec else 1
            range_val = max_val - min_val if max_val != min_val else 1.0
            quantized_ints = [
                int(round((x - min_val) / range_val * 255 - 128))
                for x in vec
            ]
            # Dequantize for error measurement
            quantized = [
                (q + 128) / 255 * range_val + min_val
                for q in quantized_ints
            ]
            quantized_bytes = len(vec)  # 1 byte each
            errors = [abs(a - b) for a, b in zip(vec, quantized)]
        else:
            return vec, QuantizationProfile(
                original_dtype="float64", quantized_dtype=target,
                original_size_bytes=original_bytes, quantized_size_bytes=original_bytes,
                compression_ratio=1.0, max_error=0.0, mean_error=0.0,
                samples_processed=len(vec),
            )

        profile = QuantizationProfile(
            original_dtype="float64",
            quantized_dtype=target,
            original_size_bytes=original_bytes,
            quantized_size_bytes=quantized_bytes,
            compression_ratio=original_bytes / max(quantized_bytes, 1),
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
            samples_processed=len(vec),
        )
        self._profiles.append(profile)
        return quantized, profile

    def _quantize_matrix(
        self, mat: list[list[float]], target: str
    ) -> tuple[list, QuantizationProfile]:
        """Quantize a 2D matrix row by row."""
        all_errors: list[float] = []
        total_original = 0
        total_quantized = 0
        quantized_rows = []

        for row in mat:
            q_row, prof = self._quantize_vector(row, target)
            quantized_rows.append(q_row)
            total_original += prof.original_size_bytes
            total_quantized += prof.quantized_size_bytes
            all_errors.append(prof.mean_error)

        profile = QuantizationProfile(
            original_dtype="float64",
            quantized_dtype=target,
            original_size_bytes=total_original,
            quantized_size_bytes=total_quantized,
            compression_ratio=total_original / max(total_quantized, 1),
            max_error=max(all_errors) if all_errors else 0.0,
            mean_error=sum(all_errors) / len(all_errors) if all_errors else 0.0,
            samples_processed=sum(p.samples_processed for p in self._profiles[-len(mat):]) if self._profiles else len(mat),
        )
        return quantized_rows, profile

    def benchmark_precision(
        self, data: list[float]
    ) -> dict[str, QuantizationProfile]:
        """Benchmark all precision levels on the same data."""
        results = {}
        for precision in ["float32", "float16", "int8"]:
            _, profile = self.quantize(data, precision)
            results[precision] = profile
        return results

    @property
    def profiles(self) -> list[QuantizationProfile]:
        return list(self._profiles)

    def __repr__(self) -> str:
        return f"Quantizer(profiles={len(self._profiles)})"
