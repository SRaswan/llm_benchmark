/// Training benchmark using Burn's `LearnerBuilder` and its built-in TUI dashboard.
///
/// What the TUI shows (live, while training runs):
///   - Loss curve (train + validation) plotted in the terminal
///   - Items/second (= training throughput, forward + backward + optimiser)
///   - Epoch / step progress bars
///   - Final summary table after `.fit()` returns
///
/// How it works:
///   Burn wraps any backend with `Autodiff<B>` to enable gradient tracking.
///   We create a random-token language-modelling dataset (same sizes as the
///   inference benchmarks), implement `TrainStep` / `ValidStep` for `Gpt`,
///   and hand everything to `LearnerBuilder`.  The TUI is the **default**
///   renderer – no extra configuration required.
///
/// Apples-to-apples vs inference (Section 1):
///   - Same `GptConfig` (tiny / small)
///   - Same batch size and sequence length
///   - Same backends (NdArray-Autodiff, WGPU-Autodiff)
///   - Random weights + random tokens → measures pure compute cost, not data loading
///   - Model load time not included in tokens/sec (identical to Section 1 methodology)

#[cfg(feature = "train")]
pub mod inner {
    use std::fmt::Display;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use burn::{
        backend::Autodiff,
        data::{
            dataloader::{batcher::Batcher, DataLoaderBuilder},
            dataset::Dataset,
        },
        nn::loss::CrossEntropyLossConfig,
        optim::AdamConfig,
        record::CompactRecorder,
        tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor, TensorData},
        train::{
            metric::{state::{FormatOptions, NumericMetricState}, LossMetric},
            ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        },
    };
    use sysinfo::System;

    use crate::model::{Gpt, GptConfig};

    /// Per-process RAM usage metric for the current process.
    pub struct ProcessMemory {
        sys: System,
        pid: sysinfo::Pid,
    }

    impl ProcessMemory {
        pub fn new() -> Self {
            Self {
                sys: System::new(),
                pid: sysinfo::Pid::from_u32(std::process::id()),
            }
        }

        fn refresh(&mut self) -> Option<u64> {
            // sysinfo reports memory in bytes.
            self.sys.refresh_processes();
            self.sys.process(self.pid).map(|p| p.memory())
        }
    }

    impl Default for ProcessMemory {
        fn default() -> Self {
            Self::new()
        }
    }

    impl burn::train::metric::Metric for ProcessMemory {
        const NAME: &'static str = "Process Memory";
        type Input = ();

        fn update(
            &mut self,
            _item: &Self::Input,
            _metadata: &burn::train::metric::MetricMetadata,
        ) -> burn::train::metric::MetricEntry {
            let mem_bytes = self.refresh().unwrap_or(0);
            let mem_mb = mem_bytes as f64 / (1024.0 * 1024.0);
            let mem_gb = mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            burn::train::metric::MetricEntry::new(
                Self::NAME.to_string(),
                format!("RSS: {:.2} MB ({:.2} GB)", mem_mb, mem_gb),
                format!("{mem_mb:.6}"),
            )
        }

        fn clear(&mut self) {}
    }

    impl burn::train::metric::Numeric for ProcessMemory {
        fn value(&self) -> f64 {
            // Best-effort numeric value; update() already refreshed.
            self.sys
                .process(self.pid)
                .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
                .unwrap_or(0.0)
        }
    }

    /// Tracks per-step wall-clock latency (train steps).
    struct StepMetricsState {
        samples_us: Vec<u64>,
        last_us: u64,
    }

    pub struct StepLatency {
        state: Arc<Mutex<StepMetricsState>>,
        last_instant: Option<Instant>,
        numeric_state: NumericMetricState,
    }

    impl StepLatency {
        pub fn new(state: Arc<Mutex<StepMetricsState>>) -> Self {
            Self {
                state,
                last_instant: None,
                numeric_state: NumericMetricState::new(),
            }
        }
    }

    impl burn::train::metric::Metric for StepLatency {
        const NAME: &'static str = "Step Latency Train";
        type Input = ();

        fn update(
            &mut self,
            _item: &Self::Input,
            _metadata: &burn::train::metric::MetricMetadata,
        ) -> burn::train::metric::MetricEntry {
            let now = Instant::now();
            let mut last_us = 0u64;
            if let Some(prev) = self.last_instant {
                last_us = now.duration_since(prev).as_micros() as u64;
                if let Ok(mut state) = self.state.lock() {
                    state.last_us = last_us;
                    state.samples_us.push(last_us);
                }
            }
            self.last_instant = Some(now);

            let n = self.state.lock().map(|v| v.samples_us.len()).unwrap_or(0);
            self.numeric_state.update(
                last_us as f64,
                1,
                FormatOptions::new(Self::NAME)
                    .unit("us")
                    .precision(0),
            )
        }

        fn clear(&mut self) {
            self.numeric_state.reset();
            self.last_instant = None;
        }
    }

    impl burn::train::metric::Numeric for StepLatency {
        fn value(&self) -> f64 {
            self.numeric_state.value()
        }
    }


    // ══════════════════════════════════════════════════════════════════════════
    // Dataset – random token sequences
    // ══════════════════════════════════════════════════════════════════════════

    /// A single language-modelling example: input tokens and next-token targets.
    #[derive(Clone, Debug)]
    pub struct LMItem {
        /// Token IDs for positions 0..seq_len
        pub input_ids: Vec<i32>,
        /// Token IDs for positions 1..seq_len+1 (next-token targets)
        pub target_ids: Vec<i32>,
    }

    /// In-memory dataset of randomly generated token sequences.
    pub struct RandomLMDataset {
        items: Vec<LMItem>,
    }

    impl RandomLMDataset {
        pub fn new(size: usize, seq_len: usize, vocab_size: usize) -> Self {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let items = (0..size)
                .map(|_| {
                    let tokens: Vec<i32> = (0..=seq_len)
                        .map(|_| rng.gen_range(0..vocab_size as i32))
                        .collect();
                    LMItem {
                        input_ids: tokens[..seq_len].to_vec(),
                        target_ids: tokens[1..=seq_len].to_vec(),
                    }
                })
                .collect();
            Self { items }
        }
    }

    impl Dataset<LMItem> for RandomLMDataset {
        fn get(&self, index: usize) -> Option<LMItem> {
            self.items.get(index).cloned()
        }

        fn len(&self) -> usize {
            self.items.len()
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Batcher – stacks individual items into a batched tensor struct
    // ══════════════════════════════════════════════════════════════════════════

    #[derive(Clone, Debug)]
    pub struct LMBatch<B: Backend> {
        pub input_ids: Tensor<B, 2, Int>,   // [batch, seq]
        pub target_ids: Tensor<B, 2, Int>,  // [batch, seq]
    }

    #[derive(Clone)]
    pub struct LMBatcher<B: Backend> {
        device: B::Device,
    }

    impl<B: Backend> LMBatcher<B> {
        pub fn new(device: B::Device) -> Self {
            Self { device }
        }
    }

    impl<B: Backend> Batcher<LMItem, LMBatch<B>> for LMBatcher<B> {
        fn batch(&self, items: Vec<LMItem>) -> LMBatch<B> {
            let batch_size = items.len();
            let seq_len = items[0].input_ids.len();

            let input_flat: Vec<i32> = items
                .iter()
                .flat_map(|i| i.input_ids.iter().copied())
                .collect();
            let target_flat: Vec<i32> = items
                .iter()
                .flat_map(|i| i.target_ids.iter().copied())
                .collect();

            let input_ids = Tensor::<B, 1, Int>::from_data(
                TensorData::new(input_flat, [batch_size * seq_len]),
                &self.device,
            )
            .reshape([batch_size, seq_len]);

            let target_ids = Tensor::<B, 1, Int>::from_data(
                TensorData::new(target_flat, [batch_size * seq_len]),
                &self.device,
            )
            .reshape([batch_size, seq_len]);

            LMBatch { input_ids, target_ids }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Training / validation steps
    // ══════════════════════════════════════════════════════════════════════════

    impl<B: AutodiffBackend> TrainStep<LMBatch<B>, ClassificationOutput<B>> for Gpt<B> {
        fn step(&self, batch: LMBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
            let output = lm_forward(self, batch);
            // `loss` is inside `output`; clone it for backward before moving into output
            let grads = output.loss.backward();
            TrainOutput::new(self, grads, output)
        }
    }

    impl<B: Backend> ValidStep<LMBatch<B>, ClassificationOutput<B>> for Gpt<B> {
        fn step(&self, batch: LMBatch<B>) -> ClassificationOutput<B> {
            lm_forward(self, batch)
        }
    }

    /// Shared forward + loss computation used by both train and valid steps.
    fn lm_forward<B: Backend>(
        model: &Gpt<B>,
        batch: LMBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.input_ids.dims();
        let logits = model.forward(batch.input_ids); // [batch, seq, vocab]
        let vocab = logits.dims()[2];

        // Flatten to [batch*seq, vocab] for cross-entropy
        let logits_flat = logits.reshape([batch_size * seq_len, vocab]);
        // Flatten targets to [batch*seq]
        let targets_flat = batch.target_ids.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Public entry point
    // ══════════════════════════════════════════════════════════════════════════

    /// Run the training benchmark with the Burn TUI dashboard.
    ///
    /// The TUI dashboard launches in-place in the terminal and shows:
    ///  - Live loss curve (train + validation)
    ///  - Training throughput (items/sec = sequences/sec)
    ///  - Epoch / step progress bars
    ///  - Summary table at the end
    pub fn run_training_benchmark<B>(
        config: &GptConfig,
        device: B::Device,
        backend_name: &str,
        num_epochs: usize,
        batch_size: usize,
    ) where
        B: AutodiffBackend,
        Gpt<B>: Display + 'static,
    {
        println!(
            "\n[ Training: {} | {} model | {} epochs | batch={} | seq={} ]",
            backend_name,
            if config.hidden_size <= 128 { "tiny" } else { "small" },
            num_epochs,
            batch_size,
            config.max_seq_len,
        );

        // ── Data ──────────────────────────────────────────────────────────────
        let seq_len = config.max_seq_len;
        let vocab_size = config.vocab_size;
        // 256 training items, 64 validation items – enough to fill several epochs
        // without the benchmark taking too long.
        let train_ds = RandomLMDataset::new(256, seq_len, vocab_size);
        let valid_ds = RandomLMDataset::new(64, seq_len, vocab_size);
        let train_steps = steps_per_epoch(train_ds.len(), batch_size);
        let valid_steps = steps_per_epoch(valid_ds.len(), batch_size);
        let total_steps = (train_steps + valid_steps) * num_epochs;

        let batcher_train = LMBatcher::<B>::new(device.clone());
        let batcher_valid = LMBatcher::<B::InnerBackend>::new(device.clone());

        let train_loader = DataLoaderBuilder::new(batcher_train)
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(1)
            .build(train_ds);

        let valid_loader = DataLoaderBuilder::new(batcher_valid)
            .batch_size(batch_size)
            .num_workers(1)
            .build(valid_ds);

        // ── Model + optimiser ─────────────────────────────────────────────────
        let model = Gpt::<B>::new(config, &device);
        let optimizer_cfg = AdamConfig::new();

        // ── LearnerBuilder – TUI dashboard is the default renderer ────────────
        // Checkpoints are written to /tmp/llm_benchmark_train/<backend>/
        let artifact_dir = format!(
            "/tmp/llm_benchmark_train/{}",
            backend_name.replace(' ', "_")
        );

        let step_state = Arc::new(Mutex::new(StepMetricsState {
            samples_us: Vec::new(),
            last_us: 0,
        }));
        let learner = LearnerBuilder::new(&artifact_dir)
            // Loss shown as a numeric chart in the TUI
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(ProcessMemory::new())
            .metric_valid_numeric(ProcessMemory::new())
            .metric_train_numeric(StepLatency::new(step_state.clone()))
            // Optional file checkpointing (comment out to skip disk writes)
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(num_epochs)
            // Print a textual summary table after .fit() returns
            .summary()
            .build(model, optimizer_cfg.clone().init::<B, Gpt<B>>(), /* lr */ 1e-4_f64);

        // ── Train – the TUI dashboard appears here ────────────────────────────
        let start = Instant::now();
        let _trained_model = learner.fit(train_loader, valid_loader);
        let total_time = start.elapsed();
        let avg_step_time = if total_steps > 0 {
            total_time / total_steps as u32
        } else {
            Duration::ZERO
        };

        println!(
            "\n  Training complete. Checkpoints written to: {}\n",
            artifact_dir
        );

        let (p50, p95) = {
            let samples = step_state.lock().ok();
            if let Some(samples) = samples {
                compute_percentiles_us(&samples.samples_us)
            } else {
                (0, 0)
            }
        };
        if p50 > 0 {
            println!("  Step latency p50/p95: {p50} / {p95} µs");
        }

        let (estimated_kernels, estimated_kernel_launch_us) =
            estimate_kernel_launch_overhead_train(config);
        let estimated_bandwidth_gbps =
            estimate_bandwidth_gbps_train(config, batch_size, seq_len, avg_step_time);
        print_training_estimates_table(
            estimated_kernels,
            estimated_kernel_launch_us,
            estimated_bandwidth_gbps,
            avg_step_time,
        );
    }

    fn steps_per_epoch(len: usize, batch_size: usize) -> usize {
        if batch_size == 0 {
            return 0;
        }
        (len + batch_size - 1) / batch_size
    }

    fn estimate_kernel_launch_overhead_train(config: &GptConfig) -> (u64, u64) {
        let per_kernel_us: u64 = std::env::var("LLM_BENCH_KERNEL_LAUNCH_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(5);
        let kernels_per_layer: u64 = 12;
        let base_kernels: u64 = 3; // embeddings + final ln + lm head
        let forward_kernels = base_kernels + kernels_per_layer * config.num_layers as u64;
        let train_kernels = forward_kernels * 3; // forward + backward + optimizer (rough)
        (train_kernels, train_kernels * per_kernel_us)
    }

    fn estimate_bandwidth_gbps_train(
        config: &GptConfig,
        batch_size: usize,
        seq_len: usize,
        avg_step_time: Duration,
    ) -> f64 {
        if avg_step_time.is_zero() {
            return 0.0;
        }
        let b = batch_size as u64;
        let s = seq_len as u64;
        let h = config.hidden_size as u64;
        let v = config.vocab_size as u64;

        let bytes_hidden = b * s * h * 4;
        let bytes_per_layer = bytes_hidden * 2; // read + write
        let bytes_logits = b * s * v * 4;
        let forward_bytes =
            bytes_hidden + bytes_per_layer * config.num_layers as u64 + bytes_logits;
        let train_bytes = forward_bytes * 3; // forward + backward + optimizer (rough)

        train_bytes as f64 / avg_step_time.as_secs_f64() / 1e9
    }

    fn print_training_estimates_table(
        estimated_kernels: u64,
        estimated_kernel_launch_us: u64,
        estimated_bandwidth_gbps: f64,
        avg_step_time: Duration,
    ) {
        println!("\n  Training Estimate Summary");
        println!("  ┌───────────────────────────────┬──────────────────────────┐");
        println!("  │ Metric                        │ Value                    │");
        println!("  ├───────────────────────────────┼──────────────────────────┤");
        println!(
            "  │ Est. kernel launches/step     │ {:>24} │",
            estimated_kernels
        );
        println!(
            "  │ Est. launch overhead/step     │ {:>21} µs │",
            estimated_kernel_launch_us
        );
        println!(
            "  │ Avg step time                 │ {:>21.2?} │",
            avg_step_time
        );
        let (bw_value, bw_unit) = format_bandwidth(estimated_bandwidth_gbps);
        println!(
            "  │ Est. bandwidth (avg step)     │ {:>19.2} {:<4} │",
            bw_value, bw_unit
        );
        println!("  └───────────────────────────────┴──────────────────────────┘\n");
    }

    fn format_bandwidth(gbps: f64) -> (f64, &'static str) {
        if gbps >= 1.0 {
            (gbps, "GB/s")
        } else {
            (gbps * 1024.0, "MB/s")
        }
    }

    fn compute_percentiles_us(values: &[u64]) -> (u64, u64) {
        if values.is_empty() {
            return (0, 0);
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        let p50 = sorted[n / 2];
        let p95 = sorted[(n as f64 * 0.95) as usize];
        (p50, p95)
    }
}

#[cfg(not(feature = "train"))]
pub mod inner {}
