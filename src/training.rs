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
            metric::LossMetric,
            ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        },
    };

    use crate::model::{Gpt, GptConfig};

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
        let optimizer = AdamConfig::new().init::<B, Gpt<B>>();

        // ── LearnerBuilder – TUI dashboard is the default renderer ────────────
        // Checkpoints are written to /tmp/llm_benchmark_train/<backend>/
        let artifact_dir = format!(
            "/tmp/llm_benchmark_train/{}",
            backend_name.replace(' ', "_")
        );

        let learner = LearnerBuilder::new(&artifact_dir)
            // Loss shown as a numeric chart in the TUI
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            // Optional file checkpointing (comment out to skip disk writes)
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(num_epochs)
            // Print a textual summary table after .fit() returns
            .summary()
            .build(model, optimizer, /* lr */ 1e-4_f64);

        // ── Train – the TUI dashboard appears here ────────────────────────────
        let _trained_model = learner.fit(train_loader, valid_loader);

        println!(
            "\n  Training complete. Checkpoints written to: {}\n",
            artifact_dir
        );
    }
}

#[cfg(not(feature = "train"))]
pub mod inner {}
