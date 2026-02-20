use std::time::{Duration, Instant};
use burn::tensor::{backend::Backend, Int, Tensor};

use crate::model::Gpt;

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
