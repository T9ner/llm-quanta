"""GGUF quantization implementation using llama.cpp."""

import subprocess
import time
from pathlib import Path
from typing import Any

from llm_quanta.quantizers.base import QuantizationResult, Quantizer


class GGUFQuantizer(Quantizer):
    """GGUF quantizer using llama.cpp tools."""

    name = "gguf"
    description = "GGUF: llama.cpp quantization format for CPU inference"

    QUANT_TYPES = {
        "q4_0": "4-bit, small model, high quality loss",
        "q4_1": "4-bit with scale, better quality",
        "q4_k_m": "4-bit K-quant medium, recommended balance",
        "q4_k_s": "4-bit K-quant small, faster",
        "q5_0": "5-bit, better quality",
        "q5_1": "5-bit with scale",
        "q5_k_m": "5-bit K-quant medium",
        "q5_k_s": "5-bit K-quant small",
        "q6_k": "6-bit K-quant, near-original quality",
        "q8_0": "8-bit, minimal quality loss",
    }

    def __init__(self, method: str = "gguf-q4"):
        super().__init__()
        self.method = method
        # Extract quant type from method name (e.g., "gguf-q4" -> "q4_k_m")
        self.quant_type = self._parse_quant_type(method)

    def _parse_quant_type(self, method: str) -> str:
        """Parse quantization type from method name."""
        if method == "gguf-q4":
            return "q4_k_m"  # Default to recommended Q4
        elif method == "gguf-q8":
            return "q8_0"
        else:
            # Try to extract quant type directly
            parts = method.split("-")
            if len(parts) > 1:
                return parts[1]
            return "q4_k_m"

    def is_available(self) -> bool:
        """Check if llama.cpp tools are available."""
        try:
            # Check for llama.cpp quantize tool
            result = subprocess.run(
                ["llama-quantize", "--help"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0 or "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
        except FileNotFoundError:
            # Also check for common alternative names
            for cmd in ["quantize", "llama.cpp-quantize"]:
                try:
                    subprocess.run([cmd, "--help"], capture_output=True)
                    return True
                except FileNotFoundError:
                    continue
            return False

    def quantize(
        self,
        model_id: str,
        output_dir: str,
        calibration_data: str = "wikitext",
        bits: int = 4,
        quant_type: str | None = None,
        ctx_size: int = 2048,
        **kwargs: Any,
    ) -> QuantizationResult:
        """Quantize a model to GGUF format.

        This requires:
        1. Converting the model to GGUF (using convert-hf-to-gguf.py)
        2. Quantizing the GGUF model (using llama-quantize)

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save GGUF model
            calibration_data: Not used for GGUF
            bits: Target bit width (used to select quant_type)
            quant_type: Specific GGUF quantization type (e.g., 'q4_k_m')
            ctx_size: Context size for conversion
        """
        if quant_type:
            self.quant_type = quant_type
        elif bits == 8:
            self.quant_type = "q8_0"
        elif bits == 5:
            self.quant_type = "q5_k_m"
        elif bits == 6:
            self.quant_type = "q6_k"
        else:
            self.quant_type = "q4_k_m"

        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get original model size estimate
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_id)
            original_size_mb = self._estimate_model_size(config)
        except Exception:
            original_size_mb = 0

        # Check if llama.cpp is available
        if not self._check_llama_cpp():
            return QuantizationResult(
                method=self.method,
                original_model=model_id,
                output_path=output_dir,
                bits=bits,
                original_size_mb=original_size_mb,
                quantized_size_mb=0,
                compression_ratio=0,
                quantization_time_seconds=0,
                success=False,
                error_message="llama.cpp not found. Install from: https://github.com/ggerganov/llama.cpp",
            )

        try:
            # Step 1: Convert to GGUF (FP16 first)
            temp_gguf = output_path / "temp_fp16.gguf"
            final_gguf = output_path / f"{model_id.split('/')[-1]}-{self.quant_type}.gguf"

            convert_result = self._convert_to_gguf(model_id, str(temp_gguf))
            if not convert_result["success"]:
                return QuantizationResult(
                    method=self.method,
                    original_model=model_id,
                    output_path=output_dir,
                    bits=bits,
                    original_size_mb=original_size_mb,
                    quantized_size_mb=0,
                    compression_ratio=0,
                    quantization_time_seconds=time.time() - start_time,
                    success=False,
                    error_message=convert_result["error"],
                )

            # Step 2: Quantize
            quant_result = self._quantize_gguf(str(temp_gguf), str(final_gguf), self.quant_type)
            if not quant_result["success"]:
                return QuantizationResult(
                    method=self.method,
                    original_model=model_id,
                    output_path=output_dir,
                    bits=bits,
                    original_size_mb=original_size_mb,
                    quantized_size_mb=0,
                    compression_ratio=0,
                    quantization_time_seconds=time.time() - start_time,
                    success=False,
                    error_message=quant_result["error"],
                )

            # Cleanup temp file
            if temp_gguf.exists():
                temp_gguf.unlink()

            quantization_time = time.time() - start_time
            quantized_size_mb = self.get_model_size_mb(final_gguf)
            compression_ratio = self.calculate_compression_ratio(original_size_mb, quantized_size_mb)

            return QuantizationResult(
                method=self.method,
                original_model=model_id,
                output_path=str(final_gguf),
                bits=bits,
                original_size_mb=original_size_mb,
                quantized_size_mb=quantized_size_mb,
                compression_ratio=compression_ratio,
                quantization_time_seconds=quantization_time,
                success=True,
                metadata={
                    "quant_type": self.quant_type,
                    "quant_description": self.QUANT_TYPES.get(self.quant_type, ""),
                    "gguf_path": str(final_gguf),
                },
            )

        except Exception as e:
            return QuantizationResult(
                method=self.method,
                original_model=model_id,
                output_path=output_dir,
                bits=bits,
                original_size_mb=original_size_mb,
                quantized_size_mb=0,
                compression_ratio=0,
                quantization_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp conversion tools are available."""
        # Check for Python conversion script or CLI tools
        try:
            import llama_cpp

            return True
        except ImportError:
            pass

        # Check for CLI tools
        for cmd in ["llama-quantize", "quantize"]:
            try:
                result = subprocess.run([cmd], capture_output=True, text=True)
                if "usage" in result.stdout.lower() or "usage" in result.stderr.lower():
                    return True
            except FileNotFoundError:
                continue

        return False

    def _convert_to_gguf(self, model_id: str, output_path: str) -> dict[str, Any]:
        """Convert HuggingFace model to GGUF format."""
        # Try using the convert script from llama.cpp
        # This is a placeholder - actual implementation would need llama.cpp installed
        return {
            "success": False,
            "error": "GGUF conversion requires llama.cpp. Install and ensure convert-hf-to-gguf.py is available.",
        }

    def _quantize_gguf(self, input_path: str, output_path: str, quant_type: str) -> dict[str, Any]:
        """Quantize a GGUF model."""
        for cmd in ["llama-quantize", "quantize"]:
            try:
                result = subprocess.run(
                    [cmd, input_path, output_path, quant_type],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return {"success": True}
                else:
                    return {"success": False, "error": result.stderr}
            except FileNotFoundError:
                continue

        return {
            "success": False,
            "error": "llama-quantize tool not found",
        }

    def _estimate_model_size(self, config: Any) -> float:
        """Estimate original model size in MB based on config."""
        if hasattr(config, "num_parameters"):
            params = config.num_parameters
        else:
            hidden_size = getattr(config, "hidden_size", 4096)
            num_layers = getattr(config, "num_hidden_layers", 32)
            intermediate_size = getattr(config, "intermediate_size", 11008)
            vocab_size = getattr(config, "vocab_size", 32000)

            params = (
                vocab_size * hidden_size
                + num_layers
                * (
                    3 * hidden_size * hidden_size
                    + hidden_size * hidden_size
                    + 2 * hidden_size * intermediate_size
                    + 2 * hidden_size
                )
            )

        return (params * 2) / (1024 * 1024)
