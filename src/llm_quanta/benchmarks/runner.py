"""Benchmark runner that orchestrates multiple benchmarks."""

import time
from dataclasses import dataclass
from typing import Any

from llm_quanta.benchmarks.accuracy import AccuracyBenchmark
from llm_quanta.benchmarks.base import Benchmark, BenchmarkResult
from llm_quanta.benchmarks.latency import LatencyBenchmark
from llm_quanta.benchmarks.memory import MemoryBenchmark
from llm_quanta.benchmarks.perplexity import PerplexityBenchmark


@dataclass
class FullBenchmarkResults:
    """Results from running all benchmarks."""

    model_path: str
    perplexity: BenchmarkResult | None = None
    latency: BenchmarkResult | None = None
    memory: BenchmarkResult | None = None
    accuracy: BenchmarkResult | None = None
    total_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_path": self.model_path,
            "perplexity": self.perplexity.value if self.perplexity else None,
            "perplexity_unit": self.perplexity.unit if self.perplexity else None,
            "latency_ms": self.latency.value if self.latency else None,
            "latency_unit": self.latency.unit if self.latency else None,
            "memory_mb": self.memory.value if self.memory else None,
            "memory_unit": self.memory.unit if self.memory else None,
            "accuracy": self.accuracy.value if self.accuracy else None,
            "accuracy_unit": self.accuracy.unit if self.accuracy else None,
            "total_time_seconds": self.total_time_seconds,
        }


class BenchmarkRunner:
    """Runner for executing benchmarks on quantized models."""

    _benchmarks: dict[str, type[Benchmark]] = {
        "perplexity": PerplexityBenchmark,
        "latency": LatencyBenchmark,
        "memory": MemoryBenchmark,
        "accuracy": AccuracyBenchmark,
    }

    def __init__(self) -> None:
        """Initialize the benchmark runner."""
        self._instances: dict[str, Benchmark] = {}

    def get_benchmark(self, name: str) -> Benchmark:
        """Get a benchmark instance by name."""
        if name not in self._benchmarks:
            available = ", ".join(self._benchmarks.keys())
            raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")

        if name not in self._instances:
            self._instances[name] = self._benchmarks[name]()

        return self._instances[name]

    def run(
        self,
        benchmark_name: str,
        model_path: str,
        test_data: str = "wikitext",
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        benchmark = self.get_benchmark(benchmark_name)
        return benchmark.run(model_path, test_data, **kwargs)

    def run_all(
        self,
        model_path: str,
        test_data: str = "wikitext",
        benchmarks: list[str] | None = None,
        **kwargs: Any,
    ) -> FullBenchmarkResults:
        """Run all specified benchmarks on a model.

        Args:
            model_path: Path to the quantized model
            test_data: Test dataset identifier
            benchmarks: List of benchmark names to run (default: all)
            **kwargs: Additional arguments passed to benchmarks

        Returns:
            FullBenchmarkResults with all benchmark values
        """
        if benchmarks is None:
            benchmarks = list(self._benchmarks.keys())

        start_time = time.time()
        results = FullBenchmarkResults(model_path=model_path)

        for name in benchmarks:
            try:
                result = self.run(name, model_path, test_data, **kwargs)
                setattr(results, name, result)
            except Exception as e:
                # Store failed result
                failed = BenchmarkResult(
                    name=name,
                    value=0,
                    unit="",
                    success=False,
                    error_message=str(e),
                )
                setattr(results, name, failed)

        results.total_time_seconds = time.time() - start_time
        return results

    def list_benchmarks(self) -> list[str]:
        """List available benchmark names."""
        return list(self._benchmarks.keys())

    def register_benchmark(self, name: str, benchmark_cls: type[Benchmark]) -> None:
        """Register a custom benchmark.

        Args:
            name: Name for the benchmark
            benchmark_cls: Benchmark class
        """
        self._benchmarks[name] = benchmark_cls
