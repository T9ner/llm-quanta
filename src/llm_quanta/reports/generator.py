"""Report generator for comparison reports."""

from typing import Any

from llm_quanta.benchmarks.runner import FullBenchmarkResults
from llm_quanta.quantizers.base import QuantizationResult
from llm_quanta.reports.comparison import ComparisonReport, MethodResults, Recommendation


class ReportGenerator:
    """Generator for comparison reports."""

    def __init__(self) -> None:
        """Initialize the report generator."""
        pass

    def generate(
        self,
        model_id: str,
        quantization_results: list[QuantizationResult],
        benchmark_results: dict[str, FullBenchmarkResults],
        include_recommendation: bool = True,
        hardware_config: dict[str, Any] | None = None,
    ) -> ComparisonReport:
        """Generate a comparison report from quantization and benchmark results.

        Args:
            model_id: The model identifier
            quantization_results: List of quantization results
            benchmark_results: Dict mapping method name to benchmark results
            include_recommendation: Whether to include a hardware recommendation
            hardware_config: Optional hardware configuration for recommendation

        Returns:
            ComparisonReport object
        """
        # Pair up quantization and benchmark results
        methods: list[MethodResults] = []
        for quant_result in quantization_results:
            bench_result = benchmark_results.get(quant_result.method)
            if bench_result:
                methods.append(
                    MethodResults(
                        quantization=quant_result,
                        benchmarks=bench_result,
                    )
                )

        # Generate recommendation if requested
        recommendation = None
        if include_recommendation and methods:
            recommendation = self._generate_recommendation(methods, hardware_config)

        return ComparisonReport(
            model_id=model_id,
            timestamp="",
            methods=methods,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        methods: list[MethodResults],
        hardware_config: dict[str, Any] | None,
    ) -> Recommendation:
        """Generate a hardware-aware recommendation.

        Args:
            methods: List of method results
            hardware_config: Hardware configuration (gpu_memory_gb, cpu_only, etc.)
        """
        # Default hardware config
        if hardware_config is None:
            hardware_config = {
                "gpu_memory_gb": 24,
                "prefer_quality": False,
                "prefer_speed": False,
                "cpu_only": False,
            }

        gpu_memory = hardware_config.get("gpu_memory_gb", 24)
        cpu_only = hardware_config.get("cpu_only", False)
        prefer_quality = hardware_config.get("prefer_quality", False)
        prefer_speed = hardware_config.get("prefer_speed", False)

        # Filter methods based on hardware constraints
        viable_methods = []
        for method in methods:
            mem_mb = method.benchmarks.memory.value if method.benchmarks.memory else 0
            mem_gb = mem_mb / 1024

            # Check GPU memory constraint
            if not cpu_only and mem_gb > gpu_memory:
                continue  # Won't fit in GPU memory

            # CPU-only: prefer GGUF
            if cpu_only and "gguf" in method.quantization.method.lower():
                viable_methods.insert(0, method)  # Prioritize GGUF for CPU
            else:
                viable_methods.append(method)

        if not viable_methods:
            viable_methods = methods  # Fallback to all methods

        # Score each method
        scored_methods = []
        for method in viable_methods:
            score = self._calculate_score(method, prefer_quality, prefer_speed)
            scored_methods.append((method, score))

        # Sort by score
        scored_methods.sort(key=lambda x: x[1]["overall"], reverse=True)

        best_method = scored_methods[0][0]
        scores = scored_methods[0][1]

        # Generate reason string
        reasons = []
        if prefer_quality:
            reasons.append("highest quality score")
        elif prefer_speed:
            reasons.append("best inference speed")
        else:
            reasons.append("best overall balance")

        if cpu_only and "gguf" in best_method.quantization.method.lower():
            reasons.append("optimized for CPU inference")

        mem_gb = (best_method.benchmarks.memory.value or 0) / 1024
        if mem_gb < gpu_memory * 0.8:
            reasons.append(f"fits well in available GPU memory ({mem_gb:.1f}GB)")

        reason = "; ".join(reasons) if reasons else "best overall performance"

        return Recommendation(
            best_method=best_method.quantization.method,
            reason=reason,
            hardware_requirements={
                "gpu_memory_gb": mem_gb,
                "cpu_only_recommended": cpu_only,
            },
            quality_score=scores["quality"],
            speed_score=scores["speed"],
            overall_score=scores["overall"],
        )

    def _calculate_score(
        self,
        method: MethodResults,
        prefer_quality: bool,
        prefer_speed: bool,
    ) -> dict[str, float]:
        """Calculate scores for a method.

        Returns:
            Dict with 'quality', 'speed', 'memory', 'overall' scores (0-1)
        """
        bench = method.benchmarks

        # Quality score (inverse perplexity)
        if bench.perplexity and bench.perplexity.success:
            quality = min(1.0, 10.0 / bench.perplexity.value)
        else:
            quality = 0.5

        # Speed score
        if bench.latency and bench.latency.success:
            speed = min(1.0, bench.latency.value / 50.0)
        else:
            speed = 0.5

        # Memory efficiency score
        if bench.memory and bench.memory.success:
            memory = min(1.0, 4000.0 / bench.memory.value)
        else:
            memory = 0.5

        # Overall score with preference weights
        if prefer_quality:
            overall = 0.6 * quality + 0.2 * speed + 0.2 * memory
        elif prefer_speed:
            overall = 0.2 * quality + 0.6 * speed + 0.2 * memory
        else:
            overall = 0.4 * quality + 0.3 * speed + 0.3 * memory

        return {
            "quality": quality,
            "speed": speed,
            "memory": memory,
            "overall": overall,
        }

    def generate_from_directory(
        self,
        model_id: str,
        output_dir: str,
        include_recommendation: bool = True,
    ) -> ComparisonReport:
        """Generate a report from a directory containing quantized models.

        Args:
            model_id: Original model identifier
            output_dir: Directory with subdirectories for each method
            include_recommendation: Whether to include recommendation

        Returns:
            ComparisonReport
        """
        from pathlib import Path

        from llm_quanta.benchmarks.runner import BenchmarkRunner
        from llm_quanta.quantizers.registry import QuantizerRegistry

        output_path = Path(output_dir)
        quantization_results = []
        benchmark_results = {}

        runner = BenchmarkRunner()

        for method_dir in output_path.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name

            # Create a placeholder quantization result
            quant_result = QuantizationResult(
                method=method_name,
                original_model=model_id,
                output_path=str(method_dir),
                bits=4,
                original_size_mb=0,
                quantized_size_mb=0,
                compression_ratio=0,
                quantization_time_seconds=0,
            )
            quantization_results.append(quant_result)

            # Run benchmarks
            bench_result = runner.run_all(str(method_dir))
            benchmark_results[method_name] = bench_result

        return self.generate(
            model_id=model_id,
            quantization_results=quantization_results,
            benchmark_results=benchmark_results,
            include_recommendation=include_recommendation,
        )
