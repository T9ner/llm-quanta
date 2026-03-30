"""Base benchmark interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    value: float
    unit: str
    higher_is_better: bool = False
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.success:
            return f"{self.name}: FAILED ({self.error_message})"
        return f"{self.name}: {self.value:.4f} {self.unit}"

    def __repr__(self) -> str:
        return f"BenchmarkResult(name={self.name!r}, value={self.value}, unit={self.unit!r})"


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    name: str = "base"
    description: str = "Base benchmark"
    unit: str = ""
    higher_is_better: bool = False

    @abstractmethod
    def run(
        self,
        model_path: str,
        test_data: str = "wikitext",
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Run the benchmark.

        Args:
            model_path: Path to the model (directory or file)
            test_data: Test dataset identifier
            **kwargs: Additional benchmark-specific arguments

        Returns:
            BenchmarkResult with the benchmark value
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load the model from the given path.

        Args:
            model_path: Path to load model from

        Returns:
            Loaded model object
        """
        pass

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """Whether this benchmark requires GPU."""
        pass
