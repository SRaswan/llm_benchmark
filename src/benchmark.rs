use std::time::{Duration, Instant};
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::model::Gpt;

// ══════════════════════════════════════════════════════════════════════════════
// Generation-level statistics (used by Candle + any autoregressive runner)
// ══════════════════════════════════════════════════════════════════════════════

/// Detailed stats for a single prompt-completion run.
///
/// # Key metrics glossary
/// | Metric | What it measures |
/// |--------|-----------------|
/// | `ttft_ms` | **Time To First Token** – how long until the 1st output token appears. Dominated by attention over the input prompt (prefill). |
/// | `output_tokens_per_sec` | **Decode throughput** – output tokens generated per wall-clock second *after* the first token. The primary apples-to-apples number. |
/// | `p50_latency_us` / `p95_latency_us` | Per-token **latency percentiles** – how long each decode step takes. |
/// | `total_time_ms` | End-to-end time including prefill + all decode steps. |
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_name: String,
    pub backend_name: String,
    /// Number of tokens in the input prompt (prefill tokens).
    pub prompt_token_count: usize,
    /// Number of tokens actually generated (output tokens).
    pub output_token_count: usize,
    /// Time to first token (prefill latency).
    pub ttft: Duration,
    /// Total time spent in the decode loop (all steps after prefill).
    pub decode_time: Duration,
    /// Raw per-step latencies in microseconds (one entry per output token).
    pub per_token_latencies_us: Vec<u64>,
    /// Median per-token latency.
    pub p50_latency_us: u64,
    /// 95th-percentile per-token latency.
    pub p95_latency_us: u64,
}

impl GenerationStats {
    pub fn new(
        prompt_name: &str,
        backend_name: &str,
        prompt_token_count: usize,
        output_token_count: usize,
        ttft: Duration,
        decode_time: Duration,
    ) -> Self {
        Self {
            prompt_name: prompt_name.to_string(),
            backend_name: backend_name.to_string(),
            prompt_token_count,
            output_token_count,
            ttft,
            decode_time,
            per_token_latencies_us: Vec::new(),
            p50_latency_us: 0,
            p95_latency_us: 0,
        }
    }

    /// Output tokens generated per second (decode phase only).
    pub fn output_tokens_per_sec(&self) -> f64 {
        if self.decode_time.is_zero() || self.output_token_count == 0 {
            return 0.0;
        }
        self.output_token_count as f64 / self.decode_time.as_secs_f64()
    }

    /// Total wall-clock time (prefill + decode).
    pub fn total_time(&self) -> Duration {
        self.ttft + self.decode_time
    }

    /// Recompute p50 / p95 from `per_token_latencies_us`. Call after
    /// populating that field.
    pub fn compute_percentiles(&mut self) {
        if self.per_token_latencies_us.is_empty() {
            return;
        }
        let mut sorted = self.per_token_latencies_us.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        self.p50_latency_us = sorted[n / 2];
        self.p95_latency_us = sorted[(n as f64 * 0.95) as usize];
    }

    pub fn print_summary(&self) {
        println!(
            "\n  ┌── Generation: {} │ {} ──",
            self.prompt_name, self.backend_name
        );
        println!(
            "  │  Prompt tokens:         {:<10}",
            self.prompt_token_count
        );
        println!(
            "  │  Output tokens:         {:<10}",
            self.output_token_count
        );
        println!(
            "  │  Time To First Token:   {:.2?}",
            self.ttft
        );
        println!(
            "  │  Decode throughput:     {:.2} output tok/s",
            self.output_tokens_per_sec()
        );
        println!(
            "  │  Total time:            {:.2?}",
            self.total_time()
        );
        if !self.per_token_latencies_us.is_empty() {
            println!(
                "  │  Per-token latency p50: {} µs",
                self.p50_latency_us
            );
            println!(
                "  │  Per-token latency p95: {} µs",
                self.p95_latency_us
            );
        }
        println!("  └─────────────────────────────────────────");
    }
}

/// Print a side-by-side comparison table for multiple `GenerationStats`.
pub fn compare_generation_results(results: &[GenerationStats]) {
    if results.is_empty() {
        return;
    }

    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        GENERATION BENCHMARK COMPARISON                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<28} │ {:<26} │ {:>8} │ {:>9} │ {:>9} ║",
        "Backend", "Prompt", "TTFT ms", "out tok/s", "p50 µs"
    );
    println!("╟─────────────────────────────┼───────────────────────────┼─────────┼──────────┼──────────╢");

    for r in results {
        println!(
            "║ {:<28} │ {:<26} │ {:>7.1} │ {:>8.1} │ {:>8} ║",
            truncate(&r.backend_name, 28),
            truncate(&r.prompt_name, 26),
            r.ttft.as_secs_f64() * 1000.0,
            r.output_tokens_per_sec(),
            r.p50_latency_us,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");

    // Find fastest decode throughput
    if let Some(fastest) = results.iter().max_by(|a, b| {
        a.output_tokens_per_sec()
            .partial_cmp(&b.output_tokens_per_sec())
            .unwrap()
    }) {
        println!(
            "\n  Best decode throughput: {} on '{}' – {:.2} output tokens/sec",
            fastest.backend_name,
            fastest.prompt_name,
            fastest.output_tokens_per_sec(),
        );
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

/// Statistics from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub backend_name: String,
    pub model_size: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub num_iterations: usize,
    pub total_time: Duration,
    pub avg_time_per_iteration: Duration,
    pub tokens_per_second: f64,
}

impl BenchmarkStats {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║          Benchmark Results: {}          ", self.backend_name);
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║ Model Size:         {:>40} ║", self.model_size);
        println!("║ Batch Size:         {:>40} ║", self.batch_size);
        println!("║ Sequence Length:    {:>40} ║", self.sequence_length);
        println!("║ Iterations:         {:>40} ║", self.num_iterations);
        println!("╟───────────────────────────────────────────────────────────╢");
        println!(
            "║ Total Time:         {:>37.2?} ║",
            self.total_time
        );
        println!(
            "║ Avg Time/Iter:      {:>37.2?} ║",
            self.avg_time_per_iteration
        );
        println!(
            "║ Throughput:         {:>34.2} tok/s ║",
            self.tokens_per_second
        );
        println!("╚═══════════════════════════════════════════════════════════╝\n");
    }
}

/// Run benchmark for a specific backend
pub fn benchmark_model<B: Backend>(
    model: &Gpt<B>,
    batch_size: usize,
    seq_len: usize,
    num_iterations: usize,
    device: &B::Device,
    backend_name: &str,
    model_size: &str,
) -> BenchmarkStats {
    println!(
        "Running benchmark: {} - {} model (batch={}, seq_len={}, iterations={})",
        backend_name, model_size, batch_size, seq_len, num_iterations
    );

    // Create random input
    let input_shape = [batch_size, seq_len];
    
    // Warmup run
    println!("  Warming up...");
    for _ in 0..3 {
        let input = Tensor::<B, 2, Int>::random(
            input_shape,
            burn::tensor::Distribution::Uniform(0.0, 100.0),
            device,
        );
        let _ = model.forward(input);
    }

    // Actual benchmark
    println!("  Benchmarking...");
    let start = Instant::now();
    
    for i in 0..num_iterations {
        let input = Tensor::<B, 2, Int>::random(
            input_shape,
            burn::tensor::Distribution::Uniform(0.0, 100.0),
            device,
        );
        let _ = model.forward(input);
        
        if (i + 1) % 10 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    
    println!(); // New line after progress dots
    let total_time = start.elapsed();
    let avg_time = total_time / num_iterations as u32;
    
    let total_tokens = batch_size * seq_len * num_iterations;
    let tokens_per_second = total_tokens as f64 / total_time.as_secs_f64();

    BenchmarkStats {
        backend_name: backend_name.to_string(),
        model_size: model_size.to_string(),
        batch_size,
        sequence_length: seq_len,
        num_iterations,
        total_time,
        avg_time_per_iteration: avg_time,
        tokens_per_second,
    }
}

/// Compare multiple benchmark results
pub fn compare_results(results: &[BenchmarkStats]) {
    if results.is_empty() {
        return;
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           BENCHMARK COMPARISON                                ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:20} │ {:10} │ {:15} │ {:15} ║",
        "Backend", "Model", "Avg Time", "Throughput"
    );
    println!("╟─────────────────────┼────────────┼─────────────────┼─────────────────╢");

    for result in results {
        println!(
            "║ {:20} │ {:10} │ {:12.2?} │ {:10.2} tok/s ║",
            result.backend_name,
            result.model_size,
            result.avg_time_per_iteration,
            result.tokens_per_second
        );
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    // Find fastest
    if let Some(fastest) = results.iter().max_by(|a, b| {
        a.tokens_per_second
            .partial_cmp(&b.tokens_per_second)
            .unwrap()
    }) {
        println!(
            "\n🏆 Fastest: {} ({:.2} tokens/second)\n",
            fastest.backend_name, fastest.tokens_per_second
        );
    }
}

/// Benchmark configuration
pub struct BenchmarkConfig {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub num_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            sequence_length: 32,
            num_iterations: 50,
        }
    }
}

impl BenchmarkConfig {
    pub fn new(batch_size: usize, sequence_length: usize, num_iterations: usize) -> Self {
        Self {
            batch_size,
            sequence_length,
            num_iterations,
        }
    }

    pub fn quick() -> Self {
        Self {
            batch_size: 2,
            sequence_length: 16,
            num_iterations: 20,
        }
    }

    pub fn thorough() -> Self {
        Self {
            batch_size: 8,
            sequence_length: 64,
            num_iterations: 100,
        }
    }
}
