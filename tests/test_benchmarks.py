"""Tests for benchmarks."""

import pytest

from llm_quanta.benchmarks.base import BenchmarkResult
from llm_quanta.benchmarks.runner import BenchmarkRunner


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_list_benchmarks(self) -> None:
        """Test listing benchmarks."""
        runner = BenchmarkRunner()
        benchmarks = runner.list_benchmarks()
        assert "perplexity" in benchmarks
        assert "latency" in benchmarks
        assert "memory" in benchmarks
        assert "accuracy" in benchmarks

    def test_get_benchmark(self) -> None:
        """Test getting a benchmark instance."""
        runner = BenchmarkRunner()
        benchmark = runner.get_benchmark("perplexity")
        assert benchmark.name == "perplexity"

    def test_get_unknown_benchmark_raises(self) -> None:
        """Test that unknown benchmark raises ValueError."""
        runner = BenchmarkRunner()
        with pytest.raises(ValueError, match="Unknown benchmark"):
            runner.get_benchmark("unknown")


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_str_representation(self) -> None:
        """Test string representation of benchmark result."""
        result = BenchmarkResult(
            name="perplexity",
            value=5.32,
            unit="perplexity",
            higher_is_better=False,
        )
        assert str(result) == "perplexity: 5.3200 perplexity"

    def test_failed_result_str(self) -> None:
        """Test string representation of failed result."""
        result = BenchmarkResult(
            name="perplexity",
            value=0,
            unit="perplexity",
            higher_is_better=False,
            success=False,
            error_message="GPU OOM",
        )
        assert "FAILED" in str(result)
        assert "GPU OOM" in str(result)
