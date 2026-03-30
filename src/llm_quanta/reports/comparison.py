"""Comparison report data structure."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from llm_quanta.benchmarks.runner import FullBenchmarkResults
from llm_quanta.quantizers.base import QuantizationResult


@dataclass
class MethodResults:
    """Results for a single quantization method."""

    quantization: QuantizationResult
    benchmarks: FullBenchmarkResults

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.quantization.method,
            "bits": self.quantization.bits,
            "original_size_mb": self.quantization.original_size_mb,
            "quantized_size_mb": self.quantization.quantized_size_mb,
            "compression_ratio": self.quantization.compression_ratio,
            "quantization_time_s": self.quantization.quantization_time_seconds,
            **self.benchmarks.to_dict(),
        }


@dataclass
class Recommendation:
    """Hardware-aware recommendation."""

    best_method: str
    reason: str
    hardware_requirements: dict[str, Any]
    quality_score: float
    speed_score: float
    overall_score: float

    def __str__(self) -> str:
        return f"Recommended: {self.best_method} ({self.reason})"


@dataclass
class ComparisonReport:
    """Full comparison report for multiple quantization methods."""

    model_id: str
    timestamp: str
    methods: list[MethodResults]
    recommendation: Recommendation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        rows = []
        for method in self.methods:
            rows.append(method.to_dict())
        return pd.DataFrame(rows)

    def get_summary_table(self) -> str:
        """Generate a markdown summary table."""
        df = self.to_dataframe()

        # Select key columns
        columns = [
            "method",
            "bits",
            "compression_ratio",
            "perplexity",
            "latency_ms",
            "memory_mb",
        ]
        available_cols = [c for c in columns if c in df.columns]
        summary_df = df[available_cols].copy()

        # Format for display
        if "perplexity" in summary_df.columns:
            summary_df["perplexity"] = summary_df["perplexity"].round(2)
        if "compression_ratio" in summary_df.columns:
            summary_df["compression_ratio"] = summary_df["compression_ratio"].round(2)
        if "latency_ms" in summary_df.columns:
            summary_df["latency_ms"] = summary_df["latency_ms"].round(2)
        if "memory_mb" in summary_df.columns:
            summary_df["memory_mb"] = summary_df["memory_mb"].round(1)

        return summary_df.to_markdown(index=False)

    def get_ranking(self, metric: str = "overall") -> list[tuple[str, float]]:
        """Rank methods by a specific metric.

        Args:
            metric: 'overall', 'quality', 'speed', 'memory'

        Returns:
            List of (method, score) tuples sorted best to worst
        """
        rankings = []

        for method in self.methods:
            bench = method.benchmarks

            if metric == "quality":
                # Lower perplexity is better
                if bench.perplexity and bench.perplexity.success:
                    score = 1.0 / bench.perplexity.value
                else:
                    score = 0
            elif metric == "speed":
                # Higher tokens/s is better
                if bench.latency and bench.latency.success:
                    score = bench.latency.value
                else:
                    score = 0
            elif metric == "memory":
                # Lower memory is better
                if bench.memory and bench.memory.success:
                    score = 1.0 / bench.memory.value
                else:
                    score = 0
            else:  # overall
                score = self._calculate_overall_score(method)

            rankings.append((method.quantization.method, score))

        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _calculate_overall_score(self, method: MethodResults) -> float:
        """Calculate an overall score balancing quality, speed, and memory."""
        bench = method.benchmarks

        # Normalize each metric (0-1 scale)
        quality_score = 0.5  # Default
        if bench.perplexity and bench.perplexity.success and bench.perplexity.value and bench.perplexity.value > 0:
            # Lower perplexity is better, normalize against baseline
            quality_score = min(1.0, 10.0 / bench.perplexity.value)  # Heuristic

        speed_score = 0.5
        if bench.latency and bench.latency.success and bench.latency.value is not None:
            # Higher tokens/s is better, normalize
            speed_score = min(1.0, bench.latency.value / 50.0)  # Heuristic

        memory_score = 0.5
        if bench.memory and bench.memory.success and bench.memory.value and bench.memory.value > 0:
            # Lower memory is better, normalize
            memory_score = min(1.0, 4000.0 / bench.memory.value)  # Heuristic

        # Weighted combination (tweak weights as needed)
        return 0.4 * quality_score + 0.3 * speed_score + 0.3 * memory_score

    def save(self, output_dir: str) -> None:
        """Save report to files.

        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save markdown report
        md_path = output_path / "report.md"
        with open(md_path, "w") as f:
            f.write(self.to_markdown())

        # Save CSV data
        csv_path = output_path / "results.csv"
        self.to_dataframe().to_csv(csv_path, index=False)

        # Save JSON for programmatic access
        import json

        json_path = output_path / "results.json"
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save HTML report
        html_path = output_path / "report.html"
        with open(html_path, "w") as f:
            f.write(self.to_html())

    def to_markdown(self) -> str:
        """Generate full markdown report."""
        lines = [
            "# LLM Quantization Comparison Report",
            "",
            f"**Model:** {self.model_id}",
            f"**Generated:** {self.timestamp}",
            "",
            "## Summary Table",
            "",
            self.get_summary_table(),
            "",
        ]

        # Add rankings
        for metric in ["overall", "quality", "speed", "memory"]:
            rankings = self.get_ranking(metric)
            lines.append(f"### {metric.capitalize()} Ranking")
            lines.append("")
            for i, (method, score) in enumerate(rankings, 1):
                lines.append(f"{i}. **{method}**: {score:.3f}")
            lines.append("")

        # Add recommendation if available
        if self.recommendation:
            lines.extend(
                [
                    "## Recommendation",
                    "",
                    f"**{self.recommendation.best_method}**",
                    "",
                    self.recommendation.reason,
                    "",
                    f"- Quality Score: {self.recommendation.quality_score:.2f}",
                    f"- Speed Score: {self.recommendation.speed_score:.2f}",
                    f"- Overall Score: {self.recommendation.overall_score:.2f}",
                    "",
                ]
            )

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML report with charts."""
        import json
        
        methods = json.dumps([m.quantization.method for m in self.methods])
        perplexity_data = json.dumps([m.benchmarks.perplexity.value if m.benchmarks.perplexity and m.benchmarks.perplexity.value is not None else 0 for m in self.methods])
        latency_data = json.dumps([m.benchmarks.latency.value if m.benchmarks.latency and m.benchmarks.latency.value is not None else 0 for m in self.methods])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Quantization Comparison - {self.model_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .chart-container {{ width: 45%; display: inline-block; margin: 20px; }}
        .recommendation {{ background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>LLM Quantization Comparison Report</h1>
    <p><strong>Model:</strong> {self.model_id}</p>
    <p><strong>Generated:</strong> {self.timestamp}</p>

    <h2>Summary Table</h2>
    {self.to_dataframe()[["method","bits","compression_ratio","quantized_size_mb"]].to_html(index=False, border=0)}

    <h2>Visualizations</h2>
    <div class="chart-container">
        <canvas id="perplexityChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="speedChart"></canvas>
    </div>

    {self._get_recommendation_html() if self.recommendation else ''}

    <script>
        const methods = {methods};
        
        // Perplexity Chart
        new Chart(document.getElementById('perplexityChart'), {{
            type: 'bar',
            data: {{
                labels: methods,
                datasets: [{{
                    label: 'Perplexity (lower is better)',
                    data: {perplexity_data},
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                }}]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true }} }} }}
        }});

        // Speed Chart
        new Chart(document.getElementById('speedChart'), {{
            type: 'bar',
            data: {{
                labels: methods,
                datasets: [{{
                    label: 'Tokens/second (higher is better)',
                    data: {latency_data},
                    backgroundColor: 'rgba(75, 192, 192, 0.5)'
                }}]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true }} }} }}
        }});
    </script>
</body>
</html>"""

    def _get_recommendation_html(self) -> str:
        if not self.recommendation:
            return ""
        return f"""
    <div class="recommendation">
        <h2>Recommendation</h2>
        <h3>{self.recommendation.best_method}</h3>
        <p>{self.recommendation.reason}</p>
    </div>
"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "methods": [m.to_dict() for m in self.methods],
            "recommendation": {
                "best_method": self.recommendation.best_method,
                "reason": self.recommendation.reason,
                "hardware_requirements": self.recommendation.hardware_requirements,
                "quality_score": self.recommendation.quality_score,
                "speed_score": self.recommendation.speed_score,
                "overall_score": self.recommendation.overall_score,
            }
            if self.recommendation
            else None,
            "metadata": self.metadata,
        }
