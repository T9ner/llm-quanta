"""Registry for quantizers."""

from llm_quanta.quantizers.base import Quantizer
from llm_quanta.quantizers.gptq import GPTQQuantizer
from llm_quanta.quantizers.awq import AWQQuantizer
from llm_quanta.quantizers.bitsandbytes import BitsAndBytesQuantizer
from llm_quanta.quantizers.gguf import GGUFQuantizer


class QuantizerRegistry:
    """Registry for available quantizers."""

    _quantizers: dict[str, type[Quantizer]] = {
        "gptq": GPTQQuantizer,
        "awq": AWQQuantizer,
        "bnb-nf4": BitsAndBytesQuantizer,
        "bnb-int8": BitsAndBytesQuantizer,
        "gguf-q4": GGUFQuantizer,
        "gguf-q8": GGUFQuantizer,
    }

    @classmethod
    def get(cls, method: str) -> Quantizer:
        """Get a quantizer instance by method name.

        Args:
            method: Quantization method name (e.g., 'gptq', 'awq')

        Returns:
            Quantizer instance

        Raises:
            ValueError: If method not found
        """
        if method not in cls._quantizers:
            available = ", ".join(cls._quantizers.keys())
            raise ValueError(f"Unknown method '{method}'. Available: {available}")

        quantizer_cls = cls._quantizers[method]
        return quantizer_cls(method=method)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered quantization methods."""
        return list(cls._quantizers.keys())

    @classmethod
    def list_installed(cls) -> list[str]:
        """List quantization methods with installed backends."""
        installed = []
        for method, quantizer_cls in cls._quantizers.items():
            quantizer = quantizer_cls(method=method)
            if quantizer.is_available():
                installed.append(method)
        return installed

    @classmethod
    def register(cls, method: str, quantizer_cls: type[Quantizer]) -> None:
        """Register a new quantizer.

        Args:
            method: Method name
            quantizer_cls: Quantizer class
        """
        cls._quantizers[method] = quantizer_cls
