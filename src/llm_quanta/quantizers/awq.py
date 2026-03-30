"""AWQ quantization implementation."""

import time
from typing import Any

from llm_quanta.quantizers.base import QuantizationResult, Quantizer


class AWQQuantizer(Quantizer):
    """AWQ (Activation-aware Weight Quantization) quantizer."""

    name = "awq"
    description = "AWQ: Activation-aware Weight Quantization for LLM Compression"

    def __init__(self, method: str = "awq"):
        super().__init__()
        self.method = method

    def is_available(self) -> bool:
        """Check if autoawq is installed."""
        try:
            import awq  # noqa: F401

            return True
        except ImportError:
            return False

    def quantize(
        self,
        model_id: str,
        output_dir: str,
        calibration_data: str = "wikitext",
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: str = "GEMM",
        **kwargs: Any,
    ) -> QuantizationResult:
        """Quantize a model using AWQ.

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save quantized model
            calibration_data: Calibration dataset name
            bits: Target bit width (4 typically)
            group_size: Group size for quantization
            zero_point: Use zero-point quantization
            version: AWQ version ('GEMM' or 'GEMV')
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
                error_message="autoawq not installed. Run: pip install autoawq",
            )

        from awq import AutoAWQForCausalLM
        from datasets import load_dataset
        from transformers import AutoTokenizer

        start_time = time.time()

        # Load tokenizer and calibration data
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Load calibration data
        if calibration_data == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            calib_data = dataset["text"][:512]
        else:
            dataset = load_dataset(calibration_data, split="train")
            calib_data = dataset["text"][:512] if "text" in dataset.features else list(dataset)[:512]

        # Get original model size
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        original_size_mb = self._estimate_model_size(config)

        # Load and quantize model
        model = AutoAWQForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": version,
        }

        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_data[:128],  # AWQ typically uses fewer samples
        )

        # Save
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)

        quantization_time = time.time() - start_time
        quantized_size_mb = self.get_model_size_mb(output_dir)
        compression_ratio = self.calculate_compression_ratio(original_size_mb, quantized_size_mb)

        return QuantizationResult(
            method=self.method,
            original_model=model_id,
            output_path=output_dir,
            bits=bits,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=compression_ratio,
            quantization_time_seconds=quantization_time,
            success=True,
            metadata={
                "group_size": group_size,
                "zero_point": zero_point,
                "version": version,
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
