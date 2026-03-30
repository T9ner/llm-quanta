"""Perplexity benchmark implementation."""

import math
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_quanta.benchmarks.base import Benchmark, BenchmarkResult


class PerplexityBenchmark(Benchmark):
    """Perplexity benchmark using WikiText or custom dataset."""

    name = "perplexity"
    description = "Calculate perplexity on a text dataset (lower is better)"
    unit = "perplexity"
    higher_is_better = False

    @property
    def requires_gpu(self) -> bool:
        return True

    def run(
        self,
        model_path: str,
        test_data: str = "wikitext",
        max_samples: int = 512,
        stride: int = 512,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Calculate perplexity on the specified dataset.

        Args:
            model_path: Path to the model
            test_data: Dataset name (default: wikitext-2)
            max_samples: Maximum number of samples to use
            stride: Stride for sliding window evaluation
        """
        try:
            model, tokenizer = self.load_model(model_path)
            device = next(model.parameters()).device

            # Load test data
            if test_data == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                texts = dataset["text"]
            else:
                dataset = load_dataset(test_data, split="test")
                texts = dataset["text"] if "text" in dataset.features else list(dataset)

            # Filter empty texts and limit samples
            texts = [t for t in texts if t.strip()][:max_samples]
            full_text = "\n\n".join(texts)

            # Tokenize
            encodings = tokenizer(full_text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)

            # Calculate perplexity using sliding window
            model.eval()
            nlls = []
            max_length = model.config.max_position_embeddings

            for i in tqdm(
                range(0, input_ids.size(1) - stride, stride),
                desc="Computing perplexity",
            ):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = i + stride
                trg_len = end_loc - i

                input_ids_slice = input_ids[:, begin_loc:end_loc]
                target_ids = input_ids_slice.clone()
                target_ids[:, :-trg_len] = -100  # Ignore positions before target

                with torch.no_grad():
                    outputs = model(input_ids_slice, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)

            # Calculate final perplexity
            total_nll = torch.stack(nlls).sum()
            ppl = torch.exp(total_nll / (input_ids.size(1) - stride))

            return BenchmarkResult(
                name=self.name,
                value=ppl.item(),
                unit=self.unit,
                higher_is_better=self.higher_is_better,
                success=True,
                metadata={
                    "test_data": test_data,
                    "max_samples": max_samples,
                    "stride": stride,
                },
            )

        except Exception as e:
            return BenchmarkResult(
                name=self.name,
                value=math.inf,
                unit=self.unit,
                higher_is_better=self.higher_is_better,
                success=False,
                error_message=str(e),
            )

    def load_model(self, model_path: str) -> tuple[Any, Any]:
        """Load model from path, detecting quantization type."""
        import json
        from pathlib import Path

        path = Path(model_path)

        # Check for BitsAndBytes config
        quant_config_path = path / "quant_config.json"
        if quant_config_path.exists():
            with open(quant_config_path) as f:
                quant_config = json.load(f)

            model_id = quant_config["model_id"]
            bits = quant_config.get("bits", 4)

            # Load with BitsAndBytes
            from transformers import BitsAndBytesConfig

            if bits == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return model, tokenizer

        # Check for GPTQ model
        if (path / "quantize_config.json").exists() or any(path.glob("*.pt")):
            try:
                from auto_gptq import AutoGPTQForCausalLM

                model = AutoGPTQForCausalLM.from_quantized(
                    path,
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(path)
                return model, tokenizer
            except ImportError:
                pass

        # Check for AWQ model
        if any(path.glob("*awq*")) or (path / "quant_config.json").exists():
            try:
                from awq import AutoAWQForCausalLM

                model = AutoAWQForCausalLM.from_quantized(path)
                tokenizer = AutoTokenizer.from_pretrained(path)
                return model, tokenizer
            except ImportError:
                pass

        # Check for GGUF model
        if path.suffix == ".gguf" or any(path.glob("*.gguf")):
            try:
                from llama_cpp import Llama

                gguf_path = path if path.suffix == ".gguf" else list(path.glob("*.gguf"))[0]
                model = Llama(str(gguf_path), n_ctx=2048, n_gpu_layers=-1)
                # GGUF models don't use transformers tokenizer
                tokenizer = None  # type: ignore
                return model, tokenizer
            except ImportError:
                raise RuntimeError("llama-cpp-python required for GGUF models")

        # Default: load as regular HF model
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
