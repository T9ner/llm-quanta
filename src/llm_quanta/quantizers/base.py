"""Base quantizer interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QuantizationResult:
    """Result of a quantization operation."""

    method: str
    original_model: str
    output_path: str
    bits: int
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantization_time_seconds: float
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Quantizer(ABC):
    """Abstract base class for quantizers."""

    name: str = "base"
    description: str = "Base quantizer"

    @abstractmethod
    def quantize(
        self,
        model_id: str,
        output_dir: str,
        calibration_data: str = "wikitext",
        bits: int = 4,
        **kwargs: Any,
    ) -> QuantizationResult:
        """Quantize a model.

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save quantized model
            calibration_data: Dataset for calibration (PTQ methods)
            bits: Target bit width
            **kwargs: Additional method-specific arguments

        Returns:
            QuantizationResult with details of the operation
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the quantization backend is installed."""
        pass

    def get_model_size_mb(self, path: str | Path) -> float:
        """Get total size of model files in MB."""
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)

        total = 0.0
        for file in path.rglob("*"):
            if file.is_file() and file.suffix in [".bin", ".safetensors", ".gguf", ".pt", ".pth"]:
                total += file.stat().st_size
        return total / (1024 * 1024)

    def calculate_compression_ratio(self, original_mb: float, quantized_mb: float) -> float:
        """Calculate compression ratio."""
        if quantized_mb == 0:
            return 0.0
        return original_mb / quantized_mb
