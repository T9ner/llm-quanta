"""BitsAndBytes quantization implementation."""

import time
from typing import Any

from llm_quanta.quantizers.base import QuantizationResult, Quantizer


class BitsAndBytesQuantizer(Quantizer):
    """BitsAndBytes quantizer for NF4 and Int8 quantization."""

    name = "bitsandbytes"
    description = "BitsAndBytes: 8-bit and 4-bit NF4 quantization"

    def __init__(self, method: str = "bnb-nf4"):
        super().__init__()
        self.method = method

    def is_available(self) -> bool:
        """Check if bitsandbytes is installed."""
        try:
            import bitsandbytes  # noqa: F401

            return True
        except ImportError:
            return False

    def quantize(
        self,
        model_id: str,
        output_dir: str,
        calibration_data: str = "wikitext",
        bits: int = 4,
        quant_type: str = "nf4",
        **kwargs: Any,
    ) -> QuantizationResult:
        """Quantize a model using BitsAndBytes.

        Note: BitsAndBytes quantization is typically done at load time,
        not saved as a separate quantized model. This implementation
        creates a wrapper that can load the model in quantized form.

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save wrapper/adapter
            calibration_data: Not used for BnB (no calibration needed)
            bits: Target bit width (4 or 8)
            quant_type: Quantization type ('nf4', 'fp4', 'int8')
        """
        if not self.is_available():
            return QuantizationResult(
                method=self.method,
                original_model=model_id,
                output_path=output_dir,
                bits=bits,
                original_size_mb=0,
                quantized_size_mb=0,
                compression_ratio=0,
                quantization_time_seconds=0,
                success=False,
                error_message="bitsandbytes not installed. Run: pip install bitsandbytes",
            )

        import json
        from pathlib import Path

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        start_time = time.time()

        # Determine quantization config based on method
        if self.method == "bnb-int8" or bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            effective_bits = 8
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            effective_bits = 4

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Load config to estimate original size
        config = AutoConfig.from_pretrained(model_id)
        original_size_mb = self._estimate_model_size(config)

        # Load model in quantized form (this is where quantization happens)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        quantization_time = time.time() - start_time

        # BitsAndBytes models are quantized in-memory; save a config wrapper
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer and a config file for loading
        tokenizer.save_pretrained(output_dir)

        wrapper_config = {
            "model_id": model_id,
            "quantization_method": self.method,
            "bits": effective_bits,
            "quant_type": quant_type if effective_bits == 4 else "int8",
            "library": "bitsandbytes",
            "load_command": f"AutoModelForCausalLM.from_pretrained('{model_id}', quantization_config=BitsAndBytesConfig(load_in_{effective_bits}bit=True))",
        }

        with open(output_path / "quant_config.json", "w") as f:
            json.dump(wrapper_config, f, indent=2)

        # Estimate quantized size (bitsandbytes quantizes in memory)
        # For 4-bit: ~0.5 bytes per param + overhead
        # For 8-bit: ~1 byte per param + overhead
        if effective_bits == 4:
            quantized_size_mb = original_size_mb / 4  # Approximate
        else:
            quantized_size_mb = original_size_mb / 2

        compression_ratio = self.calculate_compression_ratio(original_size_mb, quantized_size_mb)

        return QuantizationResult(
            method=self.method,
            original_model=model_id,
            output_path=output_dir,
            bits=effective_bits,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=compression_ratio,
            quantization_time_seconds=quantization_time,
            success=True,
            metadata={
                "quant_type": quant_type if effective_bits == 4 else "int8",
                "in_memory": True,
                "note": "BitsAndBytes quantizes at load time; use quant_config.json to reload",
            },
        )

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
