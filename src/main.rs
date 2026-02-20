mod benchmark;
mod candle_runner;
mod model;
mod prompts;
mod training;

use benchmark::{benchmark_model, compare_results, compare_generation_results, BenchmarkConfig};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, LibTorchDevice};
use model::{Gpt, GptConfig};
use prompts::{BENCHMARK_PROMPTS, WARMUP_PROMPT};

fn main() {
    println!("LLM Benchmarking Suite for Rust\n");
    println!(
        "Two benchmark sections:\n\
         1. Burn framework comparison (custom GPT, random weights) – measures\n\
            framework / backend overhead on identical architecture.\n\
         2. Candle pure-Rust LLM (real GGUF weights) – measures end-to-end\n\
            autoregressive generation on real prompts with the same metrics.\n"
    );
    print_system_info();

    // ──────────────────────────────────────────────────────────────────────────
    // Section 1 – Burn backend comparison (forward-pass throughput)
    // ──────────────────────────────────────────────────────────────────────────
    println!("{:=<80}", "");
    println!(" SECTION 1: Burn Backend Throughput (identical architecture, random weights)");
    println!("{:=<80}\n", "");
    println!(
        "Apples-to-apples: same GptConfig, same random input tensor shape,\n\
         same iteration count, same warm-up passes. Only backend differs.\n\
         Metric: input tokens processed per second (forward-pass throughput).\n"
    );

    let bench_config = BenchmarkConfig::default();
    let mut burn_results = Vec::new();

    let model_configs = vec![
        ("tiny", GptConfig::tiny()),
        ("small", GptConfig::small()),
    ];

    for (model_name, gpt_config) in &model_configs {
        println!("\n{:->60}", "");
        println!(" Model: {}  (vocab={}, hidden={}, layers={}, heads={})",
            model_name.to_uppercase(),
            gpt_config.vocab_size, gpt_config.hidden_size,
            gpt_config.num_layers, gpt_config.num_heads);
        println!("{:->60}\n", "");

        // NdArray (CPU – pure Rust ndarray)
        println!("[ NdArray (CPU) ]");
        let device_ndarray = NdArrayDevice::Cpu;
        let model_ndarray = Gpt::<NdArray>::new(gpt_config, &device_ndarray);
        let r = benchmark_model(
            &model_ndarray,
            bench_config.batch_size,
            bench_config.sequence_length,
            bench_config.num_iterations,
            &device_ndarray,
            "NdArray (CPU)",
            model_name,
        );
        r.print_summary();
        burn_results.push(r);

        // WGPU (GPU via Metal / Vulkan / DX12)
        println!("[ WGPU (GPU) ]");
        let device_wgpu = WgpuDevice::default();
        let model_wgpu = Gpt::<Wgpu>::new(gpt_config, &device_wgpu);
        let r = benchmark_model(
            &model_wgpu,
            bench_config.batch_size,
            bench_config.sequence_length,
            bench_config.num_iterations,
            &device_wgpu,
            "WGPU (GPU)",
            model_name,
        );
        r.print_summary();
        burn_results.push(r);

        // LibTorch / PyTorch (optional feature `tch`)
        #[cfg(feature = "tch")]
        {
            println!("[ LibTorch (PyTorch CPU) ]");
            let device_tch = LibTorchDevice::Cpu;
            let model_tch = Gpt::<LibTorch<f32>>::new(gpt_config, &device_tch);
            let r = benchmark_model(
                &model_tch,
                bench_config.batch_size,
                bench_config.sequence_length,
                bench_config.num_iterations,
                &device_tch,
                "LibTorch (PyTorch CPU)",
                model_name,
            );
            r.print_summary();
            burn_results.push(r);
        }
    }

    compare_results(&burn_results);

    // ──────────────────────────────────────────────────────────────────────────
    // Section 2 – Candle pure-Rust LLM (real GGUF model, real prompts)
    // ──────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        use candle_runner::inner::{CandleModelConfig, CandleRunner};

        println!("\n{:=<80}", "");
        println!(" SECTION 2: Candle Pure-Rust LLM (real weights, real prompts)");
        println!("{:=<80}\n", "");
        println!(
            "Apples-to-apples: same GGUF file, same prompts, same max_new_tokens,\n\
             greedy decoding (temperature=0). Model load time is excluded.\n\
             Metrics: TTFT (prefill latency), output tokens/sec (decode throughput),\n\
             per-token latency p50/p95.\n"
        );

        let model_config = CandleModelConfig::tiny_llama();
        println!("Loading model: {} …\n", model_config.display_name);

        match CandleRunner::load(model_config) {
            Err(e) => {
                eprintln!(
                    "  [Candle] Could not load model – skipping section 2.\n  Reason: {e}\n\
                     Tip: ensure you have internet access for the first run (model is cached\n\
                     afterwards). You can also set HUGGING_FACE_HUB_TOKEN if the repo is gated.\n"
                );
            }
            Ok(mut runner) => {
                println!(
                    "  Model loaded in {} ms (excluded from throughput numbers)\n",
                    runner.model_load_time_ms
                );

                // Warm-up pass
                println!("  Warming up with '{}' prompt …", WARMUP_PROMPT.name);
                let _ = runner.generate(&WARMUP_PROMPT);
                println!();

                let mut gen_results = Vec::new();

                for prompt in BENCHMARK_PROMPTS {
                    println!(
                        "  Prompt '{}': {} chars, max {} new tokens",
                        prompt.name,
                        prompt.text.len(),
                        prompt.max_new_tokens
                    );
                    match runner.generate(prompt) {
                        Ok(stats) => {
                            stats.print_summary();
                            gen_results.push(stats);
                        }
                        Err(e) => eprintln!("  [Candle] Generation error on '{}': {e}", prompt.name),
                    }
                }

                compare_generation_results(&gen_results);
            }
        }
    }

    #[cfg(not(feature = "candle"))]
    {
        println!("\n{:=<80}", "");
        println!(" SECTION 2: Candle Pure-Rust LLM – DISABLED");
        println!("{:=<80}", "");
        println!(
            "\n  Enable with:  cargo run --features candle\n\
             On macOS with Apple Silicon add --features candle,metal for GPU.\n"
        );
    }
    // ──────────────────────────────────────────────────────────────────────────
    // Section 3 – Training benchmark with Burn TUI dashboard
    // ──────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "train")]
    {
        use burn::backend::Autodiff;
        use training::inner::run_training_benchmark;

        println!("\n{:=<80}", "");
        println!(" SECTION 3: Training Throughput + Burn TUI Dashboard");
        println!("{:=<80}\n", "");
        println!(
            "Trains the same GPT model with Adam (random data, cross-entropy loss)\n\
             across NdArray-Autodiff (CPU) and WGPU-Autodiff (GPU).\n\
             The terminal will switch to the ratatui TUI for each backend –\n\
             it shows a live loss curve, items/sec, and epoch progress.\n"
        );

        let train_epochs = 3;
        let train_batch  = bench_config.batch_size;

        for (model_name, gpt_config) in &model_configs {
            println!("{:->60}", "");
            println!(" Model: {}  (training)", model_name.to_uppercase());
            println!("{:->60}", "");

            // NdArray + Autodiff (CPU)
            run_training_benchmark::<Autodiff<NdArray>>(
                gpt_config,
                NdArrayDevice::Cpu,
                "NdArray-Autodiff (CPU)",
                train_epochs,
                train_batch,
            );

            // WGPU + Autodiff (GPU)
            run_training_benchmark::<Autodiff<Wgpu>>(
                gpt_config,
                WgpuDevice::default(),
                "WGPU-Autodiff (GPU)",
                train_epochs,
                train_batch,
            );
        }
    }

    #[cfg(not(feature = "train"))]
    {
        println!("\n{:=<80}", "");
        println!(" SECTION 3: Training Benchmark – DISABLED");
        println!("{:=<80}", "");
        println!(
            "\n  Enable with:  cargo run --features train\n\
             Runs full forward+backward+Adam steps with the Burn TUI dashboard.\n"
        );
    }}

fn print_system_info() {
    println!("\n📊 System Information:");
    println!("  • OS: {}", std::env::consts::OS);
    println!("  • Architecture: {}", std::env::consts::ARCH);
    println!("  • Burn Version: 0.14");
    
    #[cfg(target_os = "macos")]
    println!("  • Note: WGPU on macOS uses Metal backend");
    
    println!();
}
