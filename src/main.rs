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

// ══════════════════════════════════════════════════════════════════════════════
// CLI ARGUMENT PARSING
//
// We're not pulling in a full CLI library like `clap` — that would be overkill
// for the few flags we need, and it keeps the dependency list small.  Instead
// we do a simple manual parse of std::env::args().
//
// Supported arguments:
//   --model <key>     Pick which HuggingFace model to use for Section 2.
//                     Keys: tinyllama, phi3, llama3-1b, llama3-3b
//                     Default: tinyllama
//
//   --list-models     Print available models and exit.
//
// The model key can also be set via env var:
//   LLM_BENCH_MODEL=phi3 cargo run --release --features candle
//
// Precedence: CLI arg > env var > default (tinyllama).
// ══════════════════════════════════════════════════════════════════════════════

/// Holds the parsed CLI configuration.
///
/// Right now this only has model selection, but it's a struct so we can
/// easily add more flags later (e.g. --quick, --thorough, --batch-size)
/// without changing the function signature everywhere.
struct CliConfig {
    /// Which model key the user picked (e.g. "tinyllama", "phi3").
    /// None means "use default" (tinyllama).
    model_key: Option<String>,

    /// If true, just print available models and exit.
    list_models: bool,

    /// If true, only run section 3 (training).
    section3_only: bool,
}

/// Parse command-line arguments into a CliConfig.
///
/// This is intentionally simple — just iterates through args looking for
/// known flags.  Unknown args are ignored (Cargo sometimes passes its own).
fn parse_cli() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();

    let mut config = CliConfig {
        model_key: None,
        list_models: false,
        // Check the env var that was already used in the original code
        section3_only: std::env::var("LLM_BENCH_SECTION3_ONLY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false),
    };

    // Walk through args looking for our flags.
    // We start at index 1 to skip the program name (args[0]).
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            // --model <key>
            // The next argument after --model is the model key string.
            "--model" => {
                // Make sure there's actually a value after --model
                if i + 1 < args.len() {
                    config.model_key = Some(args[i + 1].clone());
                    i += 2; // skip both --model and the key
                } else {
                    eprintln!("Error: --model requires a value (e.g. --model tinyllama)");
                    std::process::exit(1);
                }
            }
            // --list-models: print the model table and exit
            "--list-models" => {
                config.list_models = true;
                i += 1;
            }
            // Anything else: ignore (could be cargo args, etc.)
            _ => {
                i += 1;
            }
        }
    }

    // Also check env var as a fallback for model selection.
    // CLI arg takes priority if both are set.
    if config.model_key.is_none() {
        config.model_key = std::env::var("LLM_BENCH_MODEL").ok();
    }

    config
}

fn main() {
    // ── Parse CLI arguments ─────────────────────────────────────────────
    let cli = parse_cli();

    // ── Handle --list-models early exit ──────────────────────────────────
    // If the user just wants to see what models are available, print and quit.
    #[cfg(feature = "candle")]
    {
        if cli.list_models {
            candle_runner::inner::print_available_models();
            return; // exit immediately, don't run benchmarks
        }
    }
    #[cfg(not(feature = "candle"))]
    {
        if cli.list_models {
            println!("\n  Model selection requires the 'candle' feature.");
            println!("  Run with: cargo run --release --features candle -- --list-models\n");
            return;
        }
    }

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

    // Shared model sizes across sections
    let model_configs = vec![
        ("tiny", GptConfig::tiny()),
        ("small", GptConfig::small()),
    ];

    // ──────────────────────────────────────────────────────────────────────────
    // Section 1 – Burn backend comparison (forward-pass throughput)
    // ──────────────────────────────────────────────────────────────────────────
    let bench_config = BenchmarkConfig::default();
    if !cli.section3_only {
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

    // ──────────────────────────────────────────────────────────────────────────
    // Section 2 – Candle pure-Rust LLM (real GGUF model, real prompts)
    //
    // NEW: Instead of hardcoding TinyLlama, we look up the model from the
    // registry based on what the user passed via --model or LLM_BENCH_MODEL.
    // ──────────────────────────────────────────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        if cli.section3_only {
            // Skip Candle section if running training-only mode.
        } else {
        use candle_runner::inner::{CandleRunner, resolve_model, print_available_models};

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

        // ── Resolve which model to use ──────────────────────────────────
        //
        // Priority:
        //   1. --model CLI arg  (already in cli.model_key)
        //   2. LLM_BENCH_MODEL env var  (also already merged into cli.model_key)
        //   3. Default to "tinyllama"
        //
        // If the user gave a key that doesn't match anything in the
        // registry, we show the available models and bail out.
        let model_key = cli.model_key
            .as_deref()          // Option<String> → Option<&str>
            .unwrap_or("tinyllama");  // default if nothing was specified

        let model_config = match resolve_model(model_key) {
            Some(config) => config,
            None => {
                // The user typed something we don't recognise.
                // Show them what's available so they can fix it.
                eprintln!(
                    "  Error: unknown model key '{}'\n",
                    model_key
                );
                print_available_models();
                // Don't crash the whole program — just skip Section 2.
                // Section 1 results (if any) are still useful.
                eprintln!("  Skipping Section 2.\n");
                return;
            }
        };

        println!("  Selected model: {} (key: '{}')", model_config.display_name, model_key);
        println!("  Loading model: {} …\n", model_config.display_name);

        match CandleRunner::load(model_config) {
            Err(e) => {
                eprintln!(
                    "  [Candle] Could not load model – skipping section 2.\n  Reason: {e}\n\
                     Tip: ensure you have internet access for the first run (model is cached\n\
                     afterwards). You can also set HF_TOKEN if the repo is gated.\n"
                );
            }
            Ok(mut runner) => {
                println!(
                    "  Model loaded in {} ms (excluded from throughput numbers)\n",
                    runner.model_load_time_ms
                );

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

    // ──────────────────────────────────────────────────────────────────────────
    // Section 3 – Training benchmark with Burn TUI dashboard
    // ──────────────────────────────────────────────────────────────────────────
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
            run_training_benchmark::<Autodiff<Wgpu>, Gpt<Autodiff<Wgpu>>, _>(
                gpt_config,
                WgpuDevice::default(),
                "WGPU-Autodiff (GPU)",
                if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                train_epochs,
                train_batch,
                Gpt::<Autodiff<Wgpu>>::new,
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
