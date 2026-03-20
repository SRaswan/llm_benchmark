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


struct CliConfig {
    model_key: Option<String>,
    list_models: bool,
    section3_only: bool,
}

struct RuntimeConfig {
    section3_only: bool,
    candle_model: String,
    hf_token: Option<String>,
}

impl RuntimeConfig {
    fn from_cli(cli: &CliConfig) -> Self {
        let hf_token = std::env::var("LLM_BENCH_HF_TOKEN")
            .or_else(|_| std::env::var("HF_TOKEN"))
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();

        let candle_model = cli
            .model_key
            .clone()
            .unwrap_or_else(|| "tinyllama".to_string());

        Self {
            section3_only: cli.section3_only,
            candle_model,
            hf_token,
        }
    }
}


fn parse_cli() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();

    let mut config = CliConfig {
        model_key: None,
        list_models: false,
        section3_only: std::env::var("LLM_BENCH_SECTION3_ONLY")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            
            "--model" => {
                if i + 1 < args.len() {
                    config.model_key = Some(args[i + 1].clone());
                    i += 2; 
                } else {
                    eprintln!("Error: --model requires a value (e.g. --model tinyllama)");
                    std::process::exit(1);
                }
            }
            "--list-models" => {
                config.list_models = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    
    if config.model_key.is_none() {
        config.model_key = std::env::var("LLM_BENCH_MODEL").ok();
    }

    config
}

fn main() {
    let cli = parse_cli();
    let runtime = RuntimeConfig::from_cli(&cli);

    #[cfg(feature = "candle")]
    {
        if cli.list_models {
            candle_runner::inner::print_available_models();
            return; 
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

    let model_configs = vec![
        ("tiny", GptConfig::tiny()),
        ("small", GptConfig::small()),
    ];

    
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

            // WGPU (GPU via Metal / Vulkan / DX12)
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

            // LibTorch / PyTorch (optional feature `tch`)
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

    #[cfg(feature = "candle")]
    {
        if !cli.section3_only {
            use candle_runner::inner::{print_available_models, resolve_model, CandleRunner};

            println!("\n{:=<80}", "");
            println!(" SECTION 2: Candle Pure-Rust LLM (real weights, real prompts)");
            println!("{:=<80}\n", "");
            println!(
                "Apples-to-apples: same prompts, same max_new_tokens,\n\
                 greedy decoding (temperature=0). Model load time is excluded.\n\
                 Metrics: TTFT (prefill latency), output tokens/sec (decode throughput),\n\
                 per-token latency p50/p95.\n"
            );

            let model_key = cli
                .model_key
                .as_deref() 
                .unwrap_or("tinyllama"); 

            let model_config = match resolve_model(model_key) {
                Some(config) => Some(config),
                None => {
                    
                    eprintln!("  Error: unknown model key '{}'\n", model_key);
                    print_available_models();
                    eprintln!("  Skipping Section 2.\n");
                    None
                }
            };

            if let Some(model_config) = model_config {
                if let Some(token) = runtime.hf_token.as_deref() {
                    if std::env::var("HF_TOKEN").is_err()
                        && std::env::var("HUGGING_FACE_HUB_TOKEN").is_err()
                    {
                        std::env::set_var("HF_TOKEN", token);
                    }
                }

                println!(
                    "  Selected model: {} (key: '{}')",
                    model_config.display_name, model_key
                );
                println!("  Loading model: {} …\n", model_config.display_name);

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

            if train_transpose_only {
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

                run_training_benchmark::<Autodiff<NdArray>, model::GptTranspose<Autodiff<NdArray>>, _>(
                    gpt_config,
                    NdArrayDevice::Cpu,
                    "NdArray-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny-transpose" } else { "small-transpose" },
                    train_epochs,
                    train_batch,
                    model::GptTranspose::<Autodiff<NdArray>>::new,
                );

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

                run_training_benchmark::<Autodiff<NdArray>, Gpt<Autodiff<NdArray>>, _>(
                    gpt_config,
                    NdArrayDevice::Cpu,
                    "NdArray-Autodiff (CPU)",
                    if gpt_config.hidden_size <= 128 { "tiny" } else { "small" },
                    train_epochs,
                    train_batch,
                    Gpt::<Autodiff<NdArray>>::new,
                );

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
            "\n  Enable with:  cargo run --features train\n\
             Runs full forward+backward+Adam steps with the Burn TUI dashboard.\n"
        );
    }
}

fn print_system_info() {
    println!("\n📊 System Information:");
    println!("  • OS: {}", std::env::consts::OS);
    println!("  • Architecture: {}", std::env::consts::ARCH);
    println!("  • Burn Version: 0.20.1");

    #[cfg(target_os = "macos")]
    println!("  • Note: WGPU on macOS uses Metal backend");
    
    println!();
}
