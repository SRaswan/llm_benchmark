"""
Section 2 equivalent: Real LLM inference benchmark using PyTorch + HuggingFace transformers.

Runs the same prompts as the Rust/Candle Section 2, with the same greedy decoding
(temperature=0), measuring the same metrics (TTFT, decode throughput, per-token
latency p50/p95). This lets you directly compare Candle vs PyTorch on identical
workloads.

Usage:
    # Default: TinyLlama on best available device
    python3 llm_benchmark.py

    # Pick a model
    python3 llm_benchmark.py --model tinyllama
    python3 llm_benchmark.py --model llama3-1b

    # Force CPU (useful for apples-to-apples vs Candle CPU)
    python3 llm_benchmark.py --model tinyllama --device cpu

    # List available models
    python3 llm_benchmark.py --list-models

Requirements:
    pip install torch transformers
"""

import argparse
import platform
import statistics
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Model registry — mirrors the Rust candle_runner registry
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "tinyllama": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "display_name": "TinyLlama-1.1B (PyTorch)",
        "size_hint": "~2.2 GB (f16)",
        "requires_token": False,
    },
    "llama3-1b": {
        "repo": "meta-llama/Llama-3.2-1B-Instruct",
        "display_name": "Llama-3.2-1B (PyTorch)",
        "size_hint": "~2.5 GB (f16)",
        "requires_token": True,
    },
    "llama3-3b": {
        "repo": "meta-llama/Llama-3.2-3B-Instruct",
        "display_name": "Llama-3.2-3B (PyTorch)",
        "size_hint": "~6.8 GB (f16)",
        "requires_token": True,
    },
    "phi3": {
        "repo": "microsoft/Phi-3-mini-4k-instruct",
        "display_name": "Phi-3-Mini-4K (PyTorch)",
        "size_hint": "~7.6 GB (f16)",
        "requires_token": False,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Prompts — identical to src/prompts.rs in the Rust codebase
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkPrompt:
    name: str
    text: str
    max_new_tokens: int


WARMUP_PROMPT = BenchmarkPrompt(
    name="warmup",
    text="Hello",
    max_new_tokens=16,
)

BENCHMARK_PROMPTS = [
    # Short prompts: stresses decode throughput
    BenchmarkPrompt(
        name="short-creative",
        text="Write a haiku about Rust programming.",
        max_new_tokens=64,
    ),
    BenchmarkPrompt(
        name="short-factual",
        text="Explain what a transformer model is in one paragraph.",
        max_new_tokens=128,
    ),
    # Medium prompts: balanced prefill + decode
    BenchmarkPrompt(
        name="medium-code",
        text=(
            "Write a Rust function that computes the Fibonacci sequence "
            "iteratively and returns the nth number. Include error handling "
            "for negative inputs."
        ),
        max_new_tokens=256,
    ),
    BenchmarkPrompt(
        name="medium-reasoning",
        text=(
            "A train leaves city A at 60 mph. Another train leaves city B "
            "(200 miles away) at 80 mph heading toward city A. When and "
            "where do they meet? Show your work step by step."
        ),
        max_new_tokens=200,
    ),
    # Long prompt: stresses prefill (attention over many input tokens)
    BenchmarkPrompt(
        name="long-summarise",
        text=(
            "Summarise the following passage in three bullet points:\n"
            "\n"
            "The Rust programming language was originally designed by Graydon "
            "Hoare at Mozilla Research in 2006. It was first released publicly "
            "in 2010 and reached version 1.0 in May 2015. Rust is designed to "
            "be a systems programming language with a focus on three goals: "
            "safety, speed, and concurrency. It achieves memory safety without "
            "a garbage collector, instead using a system of ownership with rules "
            "that the compiler checks at compile time. The language has grown "
            "significantly in popularity, being voted the 'most loved "
            "programming language' in the Stack Overflow Developer Survey for "
            "nine consecutive years from 2016 through 2024. It is now used in "
            "the Linux kernel, Android, Windows, and major web infrastructure "
            "projects worldwide."
        ),
        max_new_tokens=150,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Generation stats — mirrors benchmark.rs GenerationStats
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationStats:
    prompt_name: str
    backend_name: str
    prompt_token_count: int
    output_token_count: int
    ttft_ms: float
    decode_time_ms: float
    per_token_latencies_ms: list = field(default_factory=list)
    p50_latency_us: float = 0.0
    p95_latency_us: float = 0.0

    @property
    def output_tokens_per_sec(self) -> float:
        if self.decode_time_ms == 0 or self.output_token_count == 0:
            return 0.0
        return self.output_token_count / (self.decode_time_ms / 1000.0)

    @property
    def total_time_ms(self) -> float:
        return self.ttft_ms + self.decode_time_ms

    def compute_percentiles(self):
        if not self.per_token_latencies_ms:
            return
        sorted_latencies = sorted(self.per_token_latencies_ms)
        n = len(sorted_latencies)
        # Convert ms to us for display (matching Rust output)
        self.p50_latency_us = sorted_latencies[n // 2] * 1000
        self.p95_latency_us = sorted_latencies[int(n * 0.95)] * 1000

    def print_summary(self):
        print(f"\n  ┌── Generation: {self.prompt_name} │ {self.backend_name} ──")
        print(f"  │  Prompt tokens:         {self.prompt_token_count:<10}")
        print(f"  │  Output tokens:         {self.output_token_count:<10}")
        if self.ttft_ms >= 1000:
            print(f"  │  Time To First Token:   {self.ttft_ms / 1000:.2f}s")
        else:
            print(f"  │  Time To First Token:   {self.ttft_ms:.2f}ms")
        print(f"  │  Decode throughput:     {self.output_tokens_per_sec:.2f} output tok/s")
        total = self.total_time_ms
        if total >= 1000:
            print(f"  │  Total time:            {total / 1000:.2f}s")
        else:
            print(f"  │  Total time:            {total:.2f}ms")
        if self.per_token_latencies_ms:
            print(f"  │  Per-token latency p50: {self.p50_latency_us:.0f} µs")
            print(f"  │  Per-token latency p95: {self.p95_latency_us:.0f} µs")
        print(f"  └─────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# Device helpers
# ══════════════════════════════════════════════════════════════════════════════

def pick_device(forced: str | None = None) -> torch.device:
    """Pick the best available device, or use the one the user forced."""
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device):
    """Block until all GPU work is done — needed for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Core generation + measurement
# ══════════════════════════════════════════════════════════════════════════════

def generate_with_stats(
    model,
    tokenizer,
    prompt: BenchmarkPrompt,
    device: torch.device,
    backend_name: str,
) -> GenerationStats:
    """
    Run greedy generation token-by-token, measuring TTFT and per-token latency.

    We do NOT use model.generate() because it doesn't give us per-token timing.
    Instead we run the autoregressive loop manually, same as the Candle runner.
    """
    # ── Tokenise ──────────────────────────────────────────────────────────
    inputs = tokenizer(prompt.text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # shape: [1, seq_len]
    prompt_token_count = input_ids.shape[1]

    # We'll build up the full sequence by appending one token at a time.
    generated_ids = input_ids.clone()

    # ── Prefill (measure TTFT) ────────────────────────────────────────────
    sync_device(device)
    prefill_start = time.perf_counter()

    # with torch.no_grad():
    #     outputs = model(generated_ids)
    #     # Greedy: pick the token with highest logit at the last position
    #     next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with torch.no_grad():
        outputs = model(generated_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync_device(device)
    ttft_ms = (time.perf_counter() - prefill_start) * 1000.0

    # Check for EOS immediately
    eos_token_id = tokenizer.eos_token_id or 2
    if next_token.item() == eos_token_id:
        return GenerationStats(
            prompt_name=prompt.name,
            backend_name=backend_name,
            prompt_token_count=prompt_token_count,
            output_token_count=0,
            ttft_ms=ttft_ms,
            decode_time_ms=0.0,
        )

    # Append the first generated token
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    output_tokens = 1
    per_token_times_ms = []

    # ── Decode loop ───────────────────────────────────────────────────────
    # Generate one token at a time, measuring each step, just like Candle.
    for _ in range(prompt.max_new_tokens - 1):
        sync_device(device)
        step_start = time.perf_counter()

        # with torch.no_grad():
        #     outputs = model(generated_ids)
        #     next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        with torch.no_grad():
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        sync_device(device)
        step_ms = (time.perf_counter() - step_start) * 1000.0
        per_token_times_ms.append(step_ms)

        if next_token.item() == eos_token_id:
            break

        # generated_ids = torch.cat([generated_ids, next_token], dim=1)
        output_tokens += 1

    decode_time_ms = sum(per_token_times_ms)

    stats = GenerationStats(
        prompt_name=prompt.name,
        backend_name=backend_name,
        prompt_token_count=prompt_token_count,
        output_token_count=output_tokens,
        ttft_ms=ttft_ms,
        decode_time_ms=decode_time_ms,
        per_token_latencies_ms=per_token_times_ms,
    )
    stats.compute_percentiles()
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table — mirrors compare_generation_results in benchmark.rs
# ══════════════════════════════════════════════════════════════════════════════

def compare_generation_results(results: list[GenerationStats]):
    if not results:
        return

    print()
    print("╔" + "═" * 90 + "╗")
    print("║" + " GENERATION BENCHMARK COMPARISON (PyTorch)".ljust(90) + "║")
    print("╠" + "═" * 90 + "╣")
    print(
        f"║ {'Backend':<28} │ {'Prompt':<26} │ "
        f"{'TTFT ms':>8} │ {'out tok/s':>9} │ {'p50 µs':>9} ║"
    )
    print("╟" + "─" * 90 + "╢")

    for r in results:
        name = r.backend_name[:28]
        prompt = r.prompt_name[:26]
        print(
            f"║ {name:<28} │ {prompt:<26} │ "
            f"{r.ttft_ms:>7.1f} │ {r.output_tokens_per_sec:>8.1f} │ "
            f"{r.p50_latency_us:>8.0f} ║"
        )

    print("╚" + "═" * 90 + "╝")

    # Find best decode throughput
    best = max(results, key=lambda r: r.output_tokens_per_sec)
    if best.output_tokens_per_sec > 0:
        print(
            f"\n  Best decode throughput: {best.backend_name} on "
            f"'{best.prompt_name}' – {best.output_tokens_per_sec:.2f} output tokens/sec"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI + main
# ══════════════════════════════════════════════════════════════════════════════

def print_available_models():
    print("\n  Available models:\n")
    print(f"  {'KEY':<14} {'DESCRIPTION':<40} {'SIZE':<14} {'TOKEN?'}")
    print(f"  {'─' * 80}")
    for key, info in MODEL_REGISTRY.items():
        print(
            f"  {key:<14} {info['display_name']:<40} "
            f"{info['size_hint']:<14} {'yes' if info['requires_token'] else 'no'}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="PyTorch LLM Inference Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="tinyllama",
        help="Model key from the registry (default: tinyllama)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a device: cpu, cuda, mps (default: auto-detect)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        return

    # ── Resolve model ─────────────────────────────────────────────────────
    if args.model not in MODEL_REGISTRY:
        print(f"\n  Error: unknown model key '{args.model}'\n")
        print_available_models()
        return

    model_info = MODEL_REGISTRY[args.model]

    # ── Print header ──────────────────────────────────────────────────────
    print("PyTorch LLM Inference Benchmark\n")
    print(f"  OS:           {platform.system().lower()}")
    print(f"  Architecture: {platform.machine()}")

    device = pick_device(args.device)
    print(f"  Device:       {device}")
    print()

    print("=" * 80)
    print(" PyTorch LLM Inference (real weights, real prompts)")
    print("=" * 80)
    print()
    print(
        "  Apples-to-apples: same prompts as Rust/Candle Section 2,\n"
        "  greedy decoding (temperature=0), model load time excluded.\n"
        "  Metrics: TTFT, output tokens/sec, per-token latency p50/p95.\n"
    )

    # ── Load model ────────────────────────────────────────────────────────
    print(f"  Loading model: {model_info['display_name']} …")

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_info["repo"])
    model = AutoModelForCausalLM.from_pretrained(
        model_info["repo"],
        dtype=torch.float16,
    ).to(device)
    model.eval()
    load_time_ms = (time.perf_counter() - load_start) * 1000

    print(f"  Model loaded in {load_time_ms:.0f} ms (excluded from throughput numbers)\n")

    backend_name = f"{model_info['display_name']}"

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"  Warming up with '{WARMUP_PROMPT.name}' prompt …")
    _ = generate_with_stats(model, tokenizer, WARMUP_PROMPT, device, backend_name)
    print()

    # ── Run benchmark prompts ─────────────────────────────────────────────
    results = []

    for prompt in BENCHMARK_PROMPTS:
        print(
            f"  Prompt '{prompt.name}': {len(prompt.text)} chars, "
            f"max {prompt.max_new_tokens} new tokens"
        )
        stats = generate_with_stats(model, tokenizer, prompt, device, backend_name)
        stats.print_summary()
        results.append(stats)

    # ── Comparison table ──────────────────────────────────────────────────
    compare_generation_results(results)


if __name__ == "__main__":
    main()