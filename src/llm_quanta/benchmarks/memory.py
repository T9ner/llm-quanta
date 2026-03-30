"""Memory benchmark implementation."""

from typing import Any

import torch

from llm_quanta.benchmarks.base import Benchmark, BenchmarkResult


class MemoryBenchmark(Benchmark):
    """Memory benchmark measuring GPU/CPU memory usage."""

    name = "memory"
    description = "Measure model memory footprint"
    unit = "MB"
    higher_is_better = False

    @property
    def requires_gpu(self) -> bool:
        return False

    def run(
        self,
        model_path: str,
        test_data: str = "wikitext",
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Measure memory usage of the model.

        Args:
            model_path: Path to the model
            test_data: Not used
        """
        try:
            model, tokenizer = self.load_model(model_path)

            # Check if GGUF model
            is_gguf = hasattr(model, "__class__") and "Llama" in model.__class__.__name__

            memory_info: dict[str, float] = {}

            # GPU memory (if available)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                # Run a small inference to capture peak memory
                if not is_gguf and tokenizer is not None:
                    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        model.generate(**inputs, max_new_tokens=10)
                    torch.cuda.synchronize()

                memory_info["gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                memory_info["gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)

            # CPU memory estimation via model parameters
            if not is_gguf:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                memory_info["param_size_mb"] = param_size

                # Buffer size (for layers, etc.)
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
                memory_info["buffer_size_mb"] = buffer_size

            # For GGUF models, use file size
            if is_gguf:
                from pathlib import Path

                gguf_path = model_path if Path(model_path).suffix == ".gguf" else str(list(Path(model_path).glob("*.gguf"))[0])
                file_size = Path(gguf_path).stat().st_size / (1024 * 1024)
                memory_info["model_file_mb"] = file_size

            # Calculate total memory estimate
            total_memory = (
                memory_info.get("gpu_allocated_mb", 0)
                or memory_info.get("param_size_mb", 0)
                or memory_info.get("model_file_mb", 0)
            )

            return BenchmarkResult(
                name=self.name,
                value=total_memory,
                unit=self.unit,
                higher_is_better=self.higher_is_better,
                success=True,
                metadata=memory_info,
            )

        except Exception as e:
            return BenchmarkResult(
                name=self.name,
                value=0,
                unit=self.unit,
                higher_is_better=self.higher_is_better,
                success=False,
                error_message=str(e),
            )

    def load_model(self, model_path: str) -> tuple[Any, Any]:
        """Load model from path."""
        from llm_quanta.benchmarks.perplexity import PerplexityBenchmark

        return PerplexityBenchmark().load_model(model_path)
