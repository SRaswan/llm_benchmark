# LLM Benchmark - Rust

A high-performance benchmarking suite for comparing LLM inference across different Burn backends and hardware targets.

## 🎯 Features

- **Multiple Backends**: Compare performance across:
  - **NdArray** (CPU backend)
  - **WGPU** (GPU backend via WebGPU/Metal/Vulkan)
  - Optional: **LibTorch** (PyTorch backend)

- **Lightweight GPT Model**: 
  - Pre-configured tiny and small transformer models
  - Easy to customize model architecture
  - Based on GPT-style decoder-only architecture

- **Comprehensive Metrics**:
  - Average inference time per iteration
  - Tokens per second throughput
  - Side-by-side comparison of backends

## 🚀 Quick Start

### Prerequisites

- Rust (latest stable)
- For GPU support: Vulkan/Metal drivers

### Build and Run

```bash
# Build the project
cargo build --release

# Run benchmarks
cargo run --release
```

## 📊 Model Configurations

### Tiny Model (Fast)
- Vocab Size: 512
- Hidden Size: 128
- Layers: 2
- Attention Heads: 2
- Max Sequence: 64

### Small Model (Balanced)
- Vocab Size: 2048
- Hidden Size: 256
- Layers: 4
- Attention Heads: 4
- Max Sequence: 128

## 🔧 Customization

### Adding Custom Model Sizes

Edit `src/model.rs` to add new configurations:

```rust
impl GptConfig {
    pub fn my_custom_model() -> Self {
        Self::new()
            .with_vocab_size(4096)
            .with_hidden_size(512)
            .with_num_layers(8)
            .with_num_heads(8)
            .with_max_seq_len(256)
            .with_intermediate_size(2048)
    }
}
```

### Adjusting Benchmark Settings

Modify the benchmark configuration in `src/main.rs`:

```rust
let bench_config = BenchmarkConfig::new(
    batch_size: 8,          // Number of sequences
    sequence_length: 64,    // Tokens per sequence
    num_iterations: 100     // Benchmark iterations
);
```

### Adding LibTorch Backend

Uncomment in `Cargo.toml`:
```toml
burn-tch = "0.14"
```

Then add to main.rs:
```rust
use burn::backend::libtorch::{LibTorch, LibTorchDevice};

let device = LibTorchDevice::Cuda(0); // or LibTorchDevice::Cpu
let model = Gpt::<LibTorch>::new(&config, &device);
```

## 📁 Project Structure

```
llm-benchmark/
├── src/
│   ├── main.rs          # Benchmark runner
│   ├── model.rs         # GPT transformer implementation
│   └── benchmark.rs     # Benchmarking utilities
├── Cargo.toml          # Dependencies
└── README.md
```

## 🎨 Example Output

```
🚀 LLM Benchmarking Suite for Rust

================================================================================
Testing TINY model configuration
================================================================================

┌─────────────────────────────────┐
│  NdArray Backend (CPU)          │
└─────────────────────────────────┘

╔═══════════════════════════════════════════════════════════╗
║          Benchmark Results: NdArray (CPU)                 ║
╠═══════════════════════════════════════════════════════════╣
║ Model Size:                                         tiny   ║
║ Batch Size:                                            4   ║
║ Sequence Length:                                      32   ║
║ Iterations:                                           50   ║
╟───────────────────────────────────────────────────────────╢
║ Total Time:                                        1.25s   ║
║ Avg Time/Iter:                                    25.0ms   ║
║ Throughput:                                  2560.00 tok/s ║
╚═══════════════════════════════════════════════════════════╝
```

## 🤝 Contributing

Feel free to add more backends, model architectures, or benchmark metrics!

## 📝 License

MIT

## 🔗 Built With

- [Burn](https://github.com/tracel-ai/burn) - Deep learning framework for Rust
