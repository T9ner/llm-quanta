"""Latency benchmark implementation."""

import statistics
import time
from typing import Any

import torch

from llm_quanta.benchmarks.base import Benchmark, BenchmarkResult


class LatencyBenchmark(Benchmark):
    """Latency benchmark measuring inference time."""

    name = "latency"
    description = "Measure inference latency (tokens/second)"
    unit = "tokens/s"
    higher_is_better = True

    # Default prompts for benchmarking
    DEFAULT_PROMPTS = [
        "The quick brown fox jumps over the lazy dog.",
        "In a surprising turn of events, scientists discovered",
        "The future of artificial intelligence depends on",
        "When implementing machine learning models, it is important to",
        "The economic impact of climate change will be",
    ]

    @property
    def requires_gpu(self) -> bool:
        return False  # Can run on CPU too

    def run(
        self,
        model_path: str,
        test_data: str = "default",
        num_samples: int = 10,
        max_new_tokens: int = 50,
        warmup: int = 2,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Measure inference latency.

        Args:
            model_path: Path to the model
            test_data: Not used (uses default prompts)
            num_samples: Number of inference runs to average
            max_new_tokens: Number of tokens to generate per run
            warmup: Number of warmup runs before timing
        """
        try:
            model, tokenizer = self.load_model(model_path)

            # Check if GGUF model (different inference API)
            is_gguf = hasattr(model, "__class__") and "Llama" in model.__class__.__name__

            prompts = self.DEFAULT_PROMPTS[:num_samples]

            # Warmup runs
            for _ in range(warmup):
                prompt = prompts[0]
                if is_gguf:
                    model(prompt, max_tokens=max_new_tokens)
                else:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Timed runs
            latencies = []
            tokens_generated = []

            for prompt in prompts:
                start_time = time.perf_counter()

                if is_gguf:
                    output = model(prompt, max_tokens=max_new_tokens)
                    gen_tokens = output["usage"]["completion_tokens"]
                else:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]

                elapsed = time.perf_counter() - start_time
                latencies.append(elapsed)
                tokens_generated.append(gen_tokens)

            # Calculate tokens per second
            total_tokens = sum(tokens_generated)
            total_time = sum(latencies)
            tokens_per_second = total_tokens / total_time

            # Also calculate per-prompt metrics
            avg_latency = statistics.mean(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

            return BenchmarkResult(
                name=self.name,
                value=tokens_per_second,
                unit=self.unit,
                higher_is_better=self.higher_is_better,
                success=True,
                metadata={
                    "num_samples": num_samples,
                    "max_new_tokens": max_new_tokens,
                    "total_tokens": total_tokens,
                    "avg_latency_ms": avg_latency * 1000,
                    "std_latency_ms": std_latency * 1000,
                },
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
        # Import from perplexity to reuse model loading logic
        from llm_quanta.benchmarks.perplexity import PerplexityBenchmark

        return PerplexityBenchmark().load_model(model_path)
