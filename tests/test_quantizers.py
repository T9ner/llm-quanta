"""Tests for quantizers."""

import pytest

from llm_quanta.quantizers.base import QuantizationResult
from llm_quanta.quantizers.registry import QuantizerRegistry


class TestQuantizerRegistry:
    """Tests for QuantizerRegistry."""

    def test_list_available(self) -> None:
        """Test listing available quantizers."""
        methods = QuantizerRegistry.list_available()
        assert "gptq" in methods
        assert "awq" in methods
        assert "bnb-nf4" in methods
        assert "gguf-q4" in methods

    def test_get_unknown_method_raises(self) -> None:
        """Test that getting unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            QuantizerRegistry.get("unknown_method")

    def test_get_gptq_quantizer(self) -> None:
        """Test getting GPTQ quantizer."""
        quantizer = QuantizerRegistry.get("gptq")
        assert quantizer.name == "gptq"

    def test_list_installed_only_returns_available(self) -> None:
        """Test that list_installed only returns methods with installed backends."""
        installed = QuantizerRegistry.list_installed()
        # This will vary based on what's installed, so just check it returns a list
        assert isinstance(installed, list)


class TestQuantizationResult:
    """Tests for QuantizationResult."""

    def test_create_result(self) -> None:
        """Test creating a quantization result."""
        result = QuantizationResult(
            method="gptq",
            original_model="test/model",
            output_path="/tmp/quantized",
            bits=4,
            original_size_mb=14000.0,
            quantized_size_mb=3500.0,
            compression_ratio=4.0,
            quantization_time_seconds=120.0,
        )
        assert result.method == "gptq"
        assert result.bits == 4
        assert result.compression_ratio == 4.0
        assert result.success is True

    def test_failed_result(self) -> None:
        """Test creating a failed result."""
        result = QuantizationResult(
            method="gptq",
            original_model="test/model",
            output_path="/tmp/quantized",
            bits=4,
            original_size_mb=0,
            quantized_size_mb=0,
            compression_ratio=0,
            quantization_time_seconds=0,
            success=False,
            error_message="Test error",
        )
        assert result.success is False
        assert result.error_message == "Test error"
