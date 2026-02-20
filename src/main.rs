mod benchmark;
mod model;

use benchmark::{benchmark_model, compare_results, BenchmarkConfig};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, LibTorchDevice};
use model::{Gpt, GptConfig};

fn main() {
    println!("🚀 LLM Benchmarking Suite for Rust\n");
    println!("This benchmark compares LLM inference performance across different Burn backends.\n");

    // Configuration
    let bench_config = BenchmarkConfig::default();
    let mut all_results = Vec::new();

    // Model configurations to test
    let model_configs = vec![
        ("tiny", GptConfig::tiny()),
        ("small", GptConfig::small()),
    ];

    for (model_name, gpt_config) in model_configs {
        println!("\n{:=>80}", "");
        println!("Testing {} model configuration", model_name.to_uppercase());
        println!("{:=>80}\n", "");

        // Benchmark on NdArray backend (CPU)
        println!("┌─────────────────────────────────┐");
        println!("│  NdArray Backend (CPU)          │");
        println!("└─────────────────────────────────┘");
        
        let device_ndarray = NdArrayDevice::Cpu;
        let model_ndarray = Gpt::<NdArray>::new(&gpt_config, &device_ndarray);
        
        let result = benchmark_model(
            &model_ndarray,
            bench_config.batch_size,
            bench_config.sequence_length,
            bench_config.num_iterations,
            &device_ndarray,
            "NdArray (CPU)",
            model_name,
        );
        result.print_summary();
        all_results.push(result);

        // Benchmark on WGPU backend (GPU)
        println!("┌─────────────────────────────────┐");
        println!("│  WGPU Backend (GPU)             │");
        println!("└─────────────────────────────────┘");
        
        let device_wgpu = WgpuDevice::default();
        let model_wgpu = Gpt::<Wgpu>::new(&gpt_config, &device_wgpu);
        
        let result = benchmark_model(
            &model_wgpu,
            bench_config.batch_size,
            bench_config.sequence_length,
            bench_config.num_iterations,
            &device_wgpu,
            "WGPU (GPU)",
            model_name,
        );
        result.print_summary();
        all_results.push(result);

        // Benchmark on LibTorch backend (PyTorch)
        #[cfg(feature = "tch")]
        {
            println!("┌─────────────────────────────────┐");
            println!("│  LibTorch Backend (PyTorch CPU) │");
            println!("└─────────────────────────────────┘");

            let device_tch = LibTorchDevice::Cpu;
            let model_tch = Gpt::<LibTorch<f32>>::new(&gpt_config, &device_tch);

            let result = benchmark_model(
                &model_tch,
                bench_config.batch_size,
                bench_config.sequence_length,
                bench_config.num_iterations,
                &device_tch,
                "LibTorch (PyTorch CPU)",
                model_name,
            );
            result.print_summary();
            all_results.push(result);
        }
    }

    // Print comparison
    compare_results(&all_results);
    
    // Additional info
    print_system_info();
}

fn print_system_info() {
    println!("\n📊 System Information:");
    println!("  • OS: {}", std::env::consts::OS);
    println!("  • Architecture: {}", std::env::consts::ARCH);
    println!("  • Burn Version: 0.14");
    
    #[cfg(target_os = "macos")]
    println!("  • Note: WGPU on macOS uses Metal backend");
    
    println!();
}
