# llm-quanta

**Unified LLM quantization with automated benchmarking and comparison.**

Quantize models with multiple methods (GPTQ, AWQ, BitsAndBytes, GGUF), benchmark them automatically, and get comparison reports with hardware-aware recommendations.

## Why This Exists

Existing tools like `auto-gptq`, `autoawq`, and `llama.cpp` each handle a single quantization method. Blog posts and papers compare methods manually. **No tool combines:**

1. Quantization with multiple methods in one command
2. Automated benchmarking (perplexity, latency, memory, task accuracy)
3. Comparison reports that tell you which method is best for your hardware

`llm-quanta` fills this gap.

## Installation

```bash
# Core package
pip install llm-quanta

# With specific quantization backends
pip install llm-quanta[gptq]      # auto-gptq
pip install llm-quanta[awq]       # autoawq
pip install llm-quanta[bitsandbytes]
pip install llm-quanta[gguf]      # llama-cpp-python

# Or install everything
pip install llm-quanta[all]
```

## Quick Start

### Quantize a model with multiple methods

```bash
llm-quanta quantize meta-llama/Llama-2-7b-hf \
  --methods gptq awq bnb-nf4 \
  --output-dir ./quantized
```

### Benchmark a quantized model

```bash
llm-quanta benchmark ./quantized/gptq \
  --benchmarks perplexity latency memory
```

### Full comparison (quantize + benchmark + report)

```bash
llm-quanta compare meta-llama/Llama-2-7b-hf \
  --methods gptq awq bnb-nf4 \
  --output-dir ./comparison \
  --recommend
```

This generates:
- `./comparison/report.html` - Visual comparison with charts
- `./comparison/report.md` - Markdown summary
- `./comparison/results.csv` - Raw benchmark data
- `./comparison/results.json` - Machine-readable results

## Supported Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `gptq` | Gradient Post-Training Quantization | GPU inference, good quality |
| `awq` | Activation-aware Weight Quantization | GPU inference, best quality |
| `bnb-nf4` | BitsAndBytes 4-bit NF4 | Quick tests, QLoRA fine-tuning |
| `bnb-int8` | BitsAndBytes 8-bit | Higher quality, more memory |
| `gguf-q4` | GGUF 4-bit K-quant | CPU inference, llama.cpp |
| `gguf-q8` | GGUF 8-bit | CPU inference, best quality |

## Python API

```python
from llm_quanta import Quantizer, BenchmarkRunner, ReportGenerator

# Quantize with a specific method
quantizer = Quantizer.get("awq")
result = quantizer.quantize("meta-llama/Llama-2-7b-hf", output_dir="./quantized")

# Run benchmarks
runner = BenchmarkRunner()
benchmarks = runner.run_all("./quantized/gptq")

# Generate comparison report
generator = ReportGenerator()
report = generator.generate(
    model_id="meta-llama/Llama-2-7b-hf",
    quantization_results=[result],
    benchmark_results={"awq": benchmarks},
)
report.save("./output")
```

## What Makes This Unique

- **Not just quantization**: Benchmarking and comparison are first-class features
- **Hardware recommendations**: "Given your 16GB GPU, use AWQ-4bit" - not blog posts
- **Calibration dataset comparison**: Test how calibration data affects quality
- **Unified interface**: Same CLI for all methods, same output format

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type check
mypy src/llm_quanta

# Format
ruff format src/
```

## License

MIT
