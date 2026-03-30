"""CLI entry point for llm-quanta."""

import click
from rich.console import Console

from llm_quanta import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="llm-quanta")
def main() -> None:
    """LLM-Quanta: Unified LLM quantization with automated benchmarking.

    Quantize models with multiple methods (GPTQ, AWQ, BitsAndBytes, GGUF),
    benchmark them automatically, and get comparison reports.
    """
    pass


@main.command()
@click.argument("model_id")
@click.option(
    "--methods",
    "-m",
    multiple=True,
    type=click.Choice(["gptq", "awq", "bnb-nf4", "bnb-int8", "gguf-q4", "gguf-q8"]),
    default=["gptq", "awq", "bnb-nf4"],
    help="Quantization methods to apply (default: gptq, awq, bnb-nf4)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./quantized",
    help="Directory to save quantized models",
)
@click.option(
    "--calibration-data",
    "-c",
    type=str,
    default="wikitext",
    help="Calibration dataset for PTQ methods",
)
@click.option(
    "--bits",
    "-b",
    type=int,
    default=4,
    help="Target bits for quantization (default: 4)",
)
def quantize(
    model_id: str,
    methods: tuple[str, ...],
    output_dir: str,
    calibration_data: str,
    bits: int,
) -> None:
    """Quantize a HuggingFace model with multiple methods.

    MODEL_ID is the HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-hf')
    """
    from llm_quanta.quantizers import QuantizerRegistry

    console.print(f"[bold blue]Quantizing {model_id}[/bold blue]")
    console.print(f"Methods: {', '.join(methods)}")
    console.print(f"Output: {output_dir}")

    results = []
    for method in methods:
        console.print(f"\n[yellow]Running {method}...[/yellow]")
        quantizer = QuantizerRegistry.get(method)
        try:
            result = quantizer.quantize(
                model_id=model_id,
                output_dir=output_dir,
                calibration_data=calibration_data,
                bits=bits,
            )
            results.append(result)
            console.print(f"[green]✓ {method} complete[/green]")
        except Exception as e:
            console.print(f"[red]✗ {method} failed: {e}[/red]")

    console.print(f"\n[bold green]Quantized {len(results)}/{len(methods)} models[/bold green]")


@main.command()
@click.argument("model_path")
@click.option(
    "--benchmarks",
    "-b",
    multiple=True,
    type=click.Choice(["perplexity", "latency", "memory", "accuracy"]),
    default=["perplexity", "latency", "memory"],
    help="Benchmarks to run",
)
@click.option(
    "--test-data",
    "-t",
    type=str,
    default="wikitext",
    help="Test dataset for benchmarks",
)
def benchmark(model_path: str, benchmarks: tuple[str, ...], test_data: str) -> None:
    """Run benchmarks on a quantized model.

    MODEL_PATH is the path to the quantized model directory.
    """
    from llm_quanta.benchmarks import BenchmarkRunner

    console.print(f"[bold blue]Benchmarking {model_path}[/bold blue]")

    runner = BenchmarkRunner()
    for bench in benchmarks:
        console.print(f"[yellow]Running {bench}...[/yellow]")
        result = runner.run(bench, model_path, test_data)
        console.print(f"[green]{bench}: {result}[/green]")


@main.command()
@click.argument("model_id")
@click.option(
    "--methods",
    "-m",
    multiple=True,
    type=click.Choice(["gptq", "awq", "bnb-nf4", "bnb-int8", "gguf-q4", "gguf-q8"]),
    default=["gptq", "awq", "bnb-nf4"],
    help="Quantization methods to compare",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./comparison",
    help="Directory to save comparison report",
)
@click.option(
    "--recommend/--no-recommend",
    default=True,
    help="Include hardware-based recommendation",
)
def compare(
    model_id: str,
    methods: tuple[str, ...],
    output_dir: str,
    recommend: bool,
) -> None:
    """Quantize a model with multiple methods, benchmark all, and generate comparison report.

    MODEL_ID is the HuggingFace model identifier.
    """
    from llm_quanta.quantizers import QuantizerRegistry
    from llm_quanta.benchmarks import BenchmarkRunner
    from llm_quanta.reports import ReportGenerator

    console.print(f"[bold blue]Full comparison for {model_id}[/bold blue]")
    console.print(f"Methods: {', '.join(methods)}")

    # Step 1: Quantize
    quantization_results = []
    for method in methods:
        console.print(f"\n[yellow]Quantizing with {method}...[/yellow]")
        quantizer = QuantizerRegistry.get(method)
        try:
            result = quantizer.quantize(model_id, output_dir=f"{output_dir}/{method}")
            quantization_results.append(result)
        except Exception as e:
            console.print(f"[red]Failed: {e}[/red]")

    # Step 2: Benchmark
    benchmark_results = {}
    runner = BenchmarkRunner()
    for result in quantization_results:
        console.print(f"[yellow]Benchmarking {result.method}...[/yellow]")
        bench_result = runner.run_all(result.output_path)
        benchmark_results[result.method] = bench_result

    # Step 3: Generate report
    console.print("\n[yellow]Generating comparison report...[/yellow]")
    generator = ReportGenerator()
    report = generator.generate(
        model_id=model_id,
        quantization_results=quantization_results,
        benchmark_results=benchmark_results,
        include_recommendation=recommend,
    )
    report.save(output_dir)

    console.print(f"\n[bold green]Report saved to {output_dir}/report.html[/bold green]")


@main.command()
@click.argument("model_path")
def info(model_path: str) -> None:
    """Show information about a quantized model."""
    console.print(f"[bold blue]Model info for {model_path}[/bold blue]")
    # TODO: Implement model info display


if __name__ == "__main__":
    main()
