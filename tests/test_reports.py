"""Tests for report generation."""

import pytest

from llm_quanta.benchmarks.base import BenchmarkResult
from llm_quanta.benchmarks.runner import FullBenchmarkResults
from llm_quanta.quantizers.base import QuantizationResult
from llm_quanta.reports.comparison import ComparisonReport, MethodResults, Recommendation
from llm_quanta.reports.generator import ReportGenerator


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_generate_basic_report(self) -> None:
        """Test generating a basic comparison report."""
        generator = ReportGenerator()

        quant_results = [
            QuantizationResult(
                method="gptq",
                original_model="test/model",
                output_path="/tmp/gptq",
                bits=4,
                original_size_mb=14000,
                quantized_size_mb=3500,
                compression_ratio=4.0,
                quantization_time_seconds=120,
            ),
            QuantizationResult(
                method="awq",
                original_model="test/model",
                output_path="/tmp/awq",
                bits=4,
                original_size_mb=14000,
                quantized_size_mb=3800,
                compression_ratio=3.68,
                quantization_time_seconds=90,
            ),
        ]

        bench_results = {
            "gptq": FullBenchmarkResults(
                model_path="/tmp/gptq",
                perplexity=BenchmarkResult(name="perplexity", value=5.69, unit="perplexity"),
                latency=BenchmarkResult(name="latency", value=45.2, unit="tokens/s"),
                memory=BenchmarkResult(name="memory", value=4500.0, unit="MB"),
            ),
            "awq": FullBenchmarkResults(
                model_path="/tmp/awq",
                perplexity=BenchmarkResult(name="perplexity", value=5.32, unit="perplexity"),
                latency=BenchmarkResult(name="latency", value=52.1, unit="tokens/s"),
                memory=BenchmarkResult(name="memory", value=4800.0, unit="MB"),
            ),
        }

        report = generator.generate(
            model_id="test/model",
            quantization_results=quant_results,
            benchmark_results=bench_results,
            include_recommendation=True,
        )

        assert report.model_id == "test/model"
        assert len(report.methods) == 2
        assert report.recommendation is not None

    def test_generate_report_without_recommendation(self) -> None:
        """Test generating report without recommendation."""
        generator = ReportGenerator()

        quant_results = [
            QuantizationResult(
                method="gptq",
                original_model="test/model",
                output_path="/tmp/gptq",
                bits=4,
                original_size_mb=14000,
                quantized_size_mb=3500,
                compression_ratio=4.0,
                quantization_time_seconds=120,
            ),
        ]

        bench_results = {
            "gptq": FullBenchmarkResults(model_path="/tmp/gptq"),
        }

        report = generator.generate(
            model_id="test/model",
            quantization_results=quant_results,
            benchmark_results=bench_results,
            include_recommendation=False,
        )

        assert report.recommendation is None


class TestComparisonReport:
    """Tests for ComparisonReport."""

    def test_to_dataframe(self) -> None:
        """Test converting report to DataFrame."""
        report = ComparisonReport(
            model_id="test/model",
            timestamp="2024-01-01T00:00:00",
            methods=[
                MethodResults(
                    quantization=QuantizationResult(
                        method="gptq",
                        original_model="test/model",
                        output_path="/tmp/gptq",
                        bits=4,
                        original_size_mb=14000,
                        quantized_size_mb=3500,
                        compression_ratio=4.0,
                        quantization_time_seconds=120,
                    ),
                    benchmarks=FullBenchmarkResults(
                        model_path="/tmp/gptq",
                        perplexity=BenchmarkResult(name="perplexity", value=5.69, unit="perplexity"),
                    ),
                ),
            ],
        )

        df = report.to_dataframe()
        assert len(df) == 1
        assert "method" in df.columns
        assert df.iloc[0]["method"] == "gptq"

    def test_get_ranking(self) -> None:
        """Test getting method rankings."""
        report = ComparisonReport(
            model_id="test/model",
            timestamp="2024-01-01T00:00:00",
            methods=[
                MethodResults(
                    quantization=QuantizationResult(
                        method="gptq",
                        original_model="test/model",
                        output_path="/tmp/gptq",
                        bits=4,
                        original_size_mb=14000,
                        quantized_size_mb=3500,
                        compression_ratio=4.0,
                        quantization_time_seconds=120,
                    ),
                    benchmarks=FullBenchmarkResults(
                        model_path="/tmp/gptq",
                        perplexity=BenchmarkResult(name="perplexity", value=5.69, unit="perplexity"),
                        latency=BenchmarkResult(name="latency", value=40.0, unit="tokens/s"),
                    ),
                ),
                MethodResults(
                    quantization=QuantizationResult(
                        method="awq",
                        original_model="test/model",
                        output_path="/tmp/awq",
                        bits=4,
                        original_size_mb=14000,
                        quantized_size_mb=3800,
                        compression_ratio=3.68,
                        quantization_time_seconds=90,
                    ),
                    benchmarks=FullBenchmarkResults(
                        model_path="/tmp/awq",
                        perplexity=BenchmarkResult(name="perplexity", value=5.32, unit="perplexity"),
                        latency=BenchmarkResult(name="latency", value=50.0, unit="tokens/s"),
                    ),
                ),
            ],
        )

        ranking = report.get_ranking("quality")
        assert len(ranking) == 2
        # AWQ should have better (lower) perplexity, so higher quality score
        assert ranking[0][0] == "awq"


class TestRecommendation:
    """Tests for Recommendation."""

    def test_str_representation(self) -> None:
        """Test string representation of recommendation."""
        rec = Recommendation(
            best_method="awq",
            reason="Best overall balance",
            hardware_requirements={"gpu_memory_gb": 4.8},
            quality_score=0.9,
            speed_score=0.8,
            overall_score=0.85,
        )
        assert "awq" in str(rec)
        assert "Best overall balance" in str(rec)
