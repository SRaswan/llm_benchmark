#![recursion_limit = "256"]

mod benchmark;
mod candle_runner;
mod model;
mod prompts;
mod training;

use benchmark::{
    benchmark_model, compare_generation_results, compare_results, BenchmarkConfig,
    GenerationStats,
};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, LibTorchDevice};
use model::{Gpt, GptConfig};
use prompts::{BENCHMARK_PROMPTS, WARMUP_PROMPT};

#[derive(Debug, Clone)]
struct RuntimeConfig {
    section3_only: bool,
    candle_model: String,
    hf_token: Option<String>,
}

impl RuntimeConfig {
    fn from_env() -> Self {
        let section3_only = std::env::var("LLM_BENCH_SECTION3_ONLY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        let candle_model =
            std::env::var("LLM_BENCH_CANDLE_MODEL").unwrap_or_else(|_| "tinyllama".to_string());

        let hf_token = std::env::var("LLM_BENCH_HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

        Self {
            section3_only,
            candle_model,
            hf_token,
        }
    }
}

#[cfg(feature = "candle")]
fn summarize_model_run(model_name: &str, stats: &[GenerationStats]) {
    if stats.is_empty() {
        println!("  No generation results for {model_name}");
        return;
    }

    let n = stats.len() as f64;

    let avg_ttft_ms = stats
        .iter()
        .map(|s| s.ttft.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / n;

    let avg_decode_tok_s = stats
        .iter()
        .map(|s| {
            if s.decode_time.is_zero() {
                0.0
            } else {
                s.output_token_count as f64 / s.decode_time.as_secs_f64()
            }
        })
        .sum::<f64>()
        / n;

    let avg_output_tokens = stats
        .iter()
        .map(|s| s.output_token_count as f64)
        .sum::<f64>()
        / n;

    println!("  Model summary:");
    println!("    • model: {model_name}");
    println!("    • avg TTFT: {:.2} ms", avg_ttft_ms);
    println!("    • avg decode throughput: {:.2} tok/s", avg_decode_tok_s);
    println!("    • avg output tokens: {:.2}", avg_output_tokens);
}

fn main() {
    let runtime = RuntimeConfig::from_env();

    println!("LLM Benchmarking Suite for Rust\n");
    println!(
        "Two benchmark sections:\n\
         1. Burn framework comparison (custom GPT, random weights) – measures\n\
            framework / backend overhead on identical architecture.\n\
         2. Candle pure-Rust LLM (real GGUF weights) – measures end-to-end\n\
            autoregressive generation on real prompts with the same metrics.\n"
    );
    print_system_info();
    print_runtime_info(&runtime);

    let model_configs = vec![("tiny", GptConfig::tiny()), ("small", GptConfig::small())];

    let bench_config = BenchmarkConfig::default();

    // -------------------------------------------------------------------------
    // Section 1 – Burn backend comparison
    // -------------------------------------------------------------------------
    if !runtime.section3_only {
        println!("{:=<80}", "");
        println!(" SECTION 1: Burn Backend Throughput (identical architecture, random weights)");
        println!("{:=<80}\n", "");
        println!(
            "Apples-to-apples: same GptConfig, same random input tensor shape,\n\
             same iteration count, same warm-up passes. Only backend differs.\n\
             Metric: input tokens processed per second (forward-pass throughput).\n"
        );

        let mut burn_results = Vec::new();

        for (model_name, gpt_config) in &model_configs {
            println!("\n{:->60}", "");
            println!(
                " Model: {}  (vocab={}, hidden={}, layers={}, heads={})",
                model_name.to_uppercase(),
                gpt_config.vocab_size,
                gpt_config.hidden_size,
                gpt_config.num_layers,
                gpt_config.num_heads
            );
            println!("{:->60}\n", "");

            println!("[ NdArray (CPU) ]");
            let device_ndarray = NdArrayDevice::Cpu;
            let model_ndarray = Gpt::<NdArray>::new(gpt_config, &device_ndarray);
            let r = benchmark_model(
                &model_ndarray,
                gpt_config,
                bench_config.batch_size,
                bench_config.sequence_length,
                bench_config.num_iterations,
                &device_ndarray,
                "NdArray (CPU)",
                model_name,
            );
            r.print_summary();
            burn_results.push(r);

            println!("[ WGPU (GPU) ]");
            let device_wgpu = WgpuDevice::default();
            let model_wgpu = Gpt::<Wgpu>::new(gpt_config, &device_wgpu);
            let r = benchmark_model(
                &model_wgpu,
                gpt_config,
                bench_config.batch_size,
                bench_config.sequence_length,
                bench_config.num_iterations,
                &device_wgpu,
                "WGPU (GPU)",
                model_name,
            );
            r.print_summary();
            burn_results.push(r);

            #[cfg(feature = "tch")]
            {
                println!("[ LibTorch (PyTorch CPU) ]");
                let device_tch = LibTorchDevice::Cpu;
                let model_tch = Gpt::<LibTorch<f32>>::new(gpt_config, &device_tch);
                let r = benchmark_model(
                    &model_tch,
                    gpt_config,
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
    }

    // -------------------------------------------------------------------------
    // Section 2 – Candle pure-Rust LLM
    // -------------------------------------------------------------------------
    #[cfg(feature = "candle")]
    {
        use candle_runner::inner::{CandleModelKind, CandleRunner};

        if !runtime.section3_only {
            println!("\n{:=<80}", "");
            println!(" SECTION 2: Candle Pure-Rust LLM (real weights, real prompts)");
            println!("{:=<80}\n", "");
            println!(
                "Apples-to-apples: same prompts, same max_new_tokens,\n\
                 greedy decoding (temperature=0). Model load time is excluded.\n\
                 Metrics: TTFT (prefill latency), output tokens/sec (decode throughput),\n\
                 per-token latency p50/p95.\n"
            );

            let models_to_run = if runtime.candle_model.eq_ignore_ascii_case("all") {
                CandleModelKind::all()
            } else {
                match CandleModelKind::parse(&runtime.candle_model) {
                    Some(m) => vec![m],
                    None => {
                        eprintln!(
                            "Unknown Candle model '{}'. Supported values: tinyllama, llama32, phi4, qwen, all",
                            runtime.candle_model
                        );
                        Vec::new()
                    }
                }
            };

            for kind in models_to_run {
                let model_config = kind.config();

                if model_config.requires_token && runtime.hf_token.is_none() {
                    eprintln!(
                        "Skipping {}: this model likely requires a Hugging Face token. \
                         Set LLM_BENCH_HF_TOKEN, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN.",
                        model_config.display_name
                    );
                    continue;
                }

                println!("\n{:->80}", "");
                println!(" Running Candle benchmark for {}", model_config.display_name);
                println!("{:->80}\n", "");

                match CandleRunner::load(model_config.clone(), runtime.hf_token.as_deref()) {
                    Err(e) => {
                        eprintln!("  [Candle] Could not load model – skipping.\n  Reason: {e}\n");
                    }
                    Ok(mut runner) => {
                        println!(
                            "  Model loaded in {} ms (excluded from throughput numbers)\n",
                            runner.model_load_time_ms
                        );

                        println!("  Warming up with '{}' prompt ...", WARMUP_PROMPT.name);
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
                                Err(e) => {
                                    eprintln!("  [Candle] Generation error on '{}': {e}", prompt.name)
                                }
                            }
                        }

                        compare_generation_results(&gen_results);
                        summarize_model_run(model_config.display_name, &gen_results);
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "candle"))]
    {
        println!("\n{:=<80}", "");
        println!(" SECTION 2: Candle Pure-Rust LLM – DISABLED");
        println!("{:=<80}", "");
        println!(
            "\n  Enable with: cargo run --features candle\n\
             On macOS with Apple Silicon add --features candle,metal for GPU.\n"
        );
    }

    // -------------------------------------------------------------------------
    // Section 3 – Training benchmark
    // -------------------------------------------------------------------------
    #[cfg(feature = "train")]
    {
        use burn::backend::Autodiff;
        use training::inner::run_training_benchmark;
        let train_transpose_only = std::env::var("LLM_BENCH_TRAIN_TRANSPOSE_ONLY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        println!("\n{:=<80}", "");
        println!(" SECTION 3: Training Throughput + Burn TUI Dashboard");
        println!("{:=<80}\n", "");
        println!(
            "Trains the same GPT model with Adam (random data, cross-entropy loss)\n\
             across NdArray-Autodiff (CPU) and WGPU-Autodiff (GPU).\n\
             The terminal will switch to the ratatui TUI for each backend –\n\
             it shows a live loss curve, items/sec, and epoch progress.\n\
             Tip: set LLM_BENCH_TRAIN_TRANSPOSE_ONLY=1 to run only the transpose-tied model.\n"
        );

        let train_epochs = 3;
        let train_batch = bench_config.batch_size;

        for (model_name, gpt_config) in &model_configs {
            let train_label = if train_transpose_only {
                format!("{}-TRANSPOSE", model_name.to_uppercase())
            } else {
                model_name.to_uppercase()
            };
            println!("{:->60}", "");
            println!(" Model: {}  (training)", train_label);
            println!("{:->60}", "");

            if train_transpose_only {
                // LibTorch + Autodiff (CPU) - transpose GPT (run first)
                #[cfg(feature = "tch")]
                run_training_benchmark::<Autodiff<LibTorch<f32>>, model::GptTranspose<Autodiff<LibTorch<f32>>>, _>(
                    gpt_config,
                    LibTorchDevice::Cpu,
                    "LibTorch-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny-transpose" } else { "small-transpose" },
                    train_epochs,
                    train_batch,
                    model::GptTranspose::<Autodiff<LibTorch<f32>>>::new,
                );

                // NdArray + Autodiff (CPU) - transpose GPT
                run_training_benchmark::<Autodiff<NdArray>, model::GptTranspose<Autodiff<NdArray>>, _>(
                    gpt_config,
                    NdArrayDevice::Cpu,
                    "NdArray-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny-transpose" } else { "small-transpose" },
                    train_epochs,
                    train_batch,
                    model::GptTranspose::<Autodiff<NdArray>>::new,
                );

                // WGPU + Autodiff (GPU) - transpose GPT
                run_training_benchmark::<Autodiff<Wgpu>, model::GptTranspose<Autodiff<Wgpu>>, _>(
                    gpt_config,
                    WgpuDevice::default(),
                    "WGPU-Autodiff (GPU)",
                    if gpt_config.hidden_size <= 128 { "tiny-transpose" } else { "small-transpose" },
                    train_epochs,
                    train_batch,
                    model::GptTranspose::<Autodiff<Wgpu>>::new,
                );
            } else {
                // LibTorch + Autodiff (CPU) - standard GPT (run first)
                #[cfg(feature = "tch")]
                run_training_benchmark::<Autodiff<LibTorch<f32>>, Gpt<Autodiff<LibTorch<f32>>>, _>(
                    gpt_config,
                    LibTorchDevice::Cpu,
                    "LibTorch-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                    train_epochs,
                    train_batch,
                    Gpt::<Autodiff<LibTorch<f32>>>::new,
                );

                // NdArray + Autodiff (CPU) - standard GPT
                run_training_benchmark::<Autodiff<NdArray>, Gpt<Autodiff<NdArray>>, _>(
                    gpt_config,
                    NdArrayDevice::Cpu,
                    "NdArray-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                    train_epochs,
                    train_batch,
                    Gpt::<Autodiff<NdArray>>::new,
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
            "\n  Enable with: cargo run --features train\n\
             Runs full forward+backward+Adam steps with the Burn TUI dashboard.\n"
        );
    }
}
}

fn print_system_info() {
    println!("\nSystem Information:");
    println!("  • OS: {}", std::env::consts::OS);
    println!("  • Architecture: {}", std::env::consts::ARCH);
    println!("  • Burn Version: 0.14");

    #[cfg(target_os = "macos")]
    println!("  • Note: WGPU on macOS uses Metal backend");

    println!();
}

fn print_runtime_info(runtime: &RuntimeConfig) {
    println!("Runtime config:");
    println!("  • section3_only: {}", runtime.section3_only);
    println!("  • candle model selection: {}", runtime.candle_model);
    println!(
        "  • HF token: {}",
        if runtime.hf_token.is_some() {
            "provided"
        } else {
            "not provided"
        }
    );
    println!();
}