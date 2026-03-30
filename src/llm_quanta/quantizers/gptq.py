"""GPTQ quantization implementation."""

import time
from typing import Any

from llm_quanta.quantizers.base import QuantizationResult, Quantizer


class GPTQQuantizer(Quantizer):
    """GPTQ (Gradient Post-Training Quantization) quantizer."""

    name = "gptq"
    description = "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

    def __init__(self, method: str = "gptq"):
        super().__init__()
        self.method = method

    def is_available(self) -> bool:
        """Check if auto-gptq is installed."""
        try:
            import auto_gptq  # noqa: F401

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
        desc_act: bool = True,
        sym: bool = False,
        true_sequential: bool = True,
        **kwargs: Any,
    ) -> QuantizationResult:
        """Quantize a model using GPTQ.

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save quantized model
            calibration_data: Calibration dataset name
            bits: Target bit width (2, 3, 4, 8)
            group_size: Group size for quantization
            desc_act: Use desc_act (activation-order quantization)
            sym: Use symmetric quantization
            true_sequential: Use true sequential quantization
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
                error_message="auto-gptq not installed. Run: pip install auto-gptq",
            )

        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from datasets import load_dataset
        from transformers import AutoTokenizer

        start_time = time.time()

        # Load tokenizer and calibration data
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if calibration_data == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            calib_data = dataset["text"][:512]  # Use 512 samples for calibration
        else:
            dataset = load_dataset(calibration_data, split="train")
            calib_data = dataset["text"][:512] if "text" in dataset.features else list(dataset)[:512]

        # Get original model size estimate (approximate from config)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        original_size_mb = self._estimate_model_size(config)

        # Quantize
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            true_sequential=true_sequential,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            model_id,
            quantize_config,
            trust_remote_code=True,
        )

        model.quantize(calib_data)

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
                "desc_act": desc_act,
                "sym": sym,
                "true_sequential": true_sequential,
            },
        )

    def _estimate_model_size(self, config: Any) -> float:
        """Estimate original model size in MB based on config."""
        # Approximate: params * 2 bytes (FP16)
        if hasattr(config, "num_parameters"):
            params = config.num_parameters
        else:
            # Estimate from hidden_size, num_layers, etc.
            hidden_size = getattr(config, "hidden_size", 4096)
            num_layers = getattr(config, "num_hidden_layers", 32)
            intermediate_size = getattr(config, "intermediate_size", 11008)
            vocab_size = getattr(config, "vocab_size", 32000)

            # Rough parameter count for LLaMA-style models
            params = (
                vocab_size * hidden_size  # Embedding
                + num_layers
                * (
                    3 * hidden_size * hidden_size  # Attention (Q, K, V)
                    + hidden_size * hidden_size  # Output projection
                    + 2 * hidden_size * intermediate_size  # MLP
                    + 2 * hidden_size  # Layer norms
                )
            )

        return (params * 2) / (1024 * 1024)  # FP16 = 2 bytes per param
