/// Pure-Rust LLM inference using HuggingFace Candle.
///
/// This module loads a GGUF-quantised model (default: TinyLlama-1.1B-Q4_K_M)
/// from the HuggingFace Hub and runs autoregressive generation, measuring the
/// same metrics as the Burn benchmarks so the results can be directly compared.
///
/// # Apples-to-apples notes
/// • The same GGUF file is used here and, via `burn-tch`, by the LibTorch
///   backend when feature `tch` is also enabled.
/// • Temperature is always 0 (greedy), eliminating sampling randomness.
/// • Model load time is measured separately and excluded from throughput.
/// • We measure *output* tokens / second (the number that actually required
///   a forward pass through the decoder), not the input (prefilled) tokens.

#[cfg(feature = "candle")]
pub mod inner {
    use std::path::PathBuf;
    use std::time::Instant;

    use candle_core::{Device, Tensor};
    use candle_transformers::models::quantized_llama::ModelWeights;
    use candle_transformers::generation::LogitsProcessor;
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use tokenizers::Tokenizer;

    use crate::benchmark::GenerationStats;
    use crate::prompts::BenchmarkPrompt;

    // ──────────────────────────────────────────────────────────────────────────
    // Model configuration
    // ──────────────────────────────────────────────────────────────────────────

    /// Which GGUF model to download and benchmark.
    #[derive(Debug, Clone)]
    pub struct CandleModelConfig {
        /// HuggingFace repo id for the GGUF weights.
        pub hf_repo: &'static str,
        /// Filename of the GGUF file inside that repo.
        pub gguf_file: &'static str,
        /// HuggingFace repo id for the tokeniser (usually the original model).
        pub tokenizer_repo: &'static str,
        /// Human-readable name displayed in result tables.
        pub display_name: &'static str,
    }

    impl CandleModelConfig {
        /// TinyLlama 1.1B – tiny enough to run on CPU in reasonable time.
        pub fn tiny_llama() -> Self {
            Self {
                hf_repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                tokenizer_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                display_name: "TinyLlama-1.1B Q4_K_M (Candle)",
            }
        }

        /// Phi-3 Mini 4K – small but stronger than TinyLlama.
        pub fn phi3_mini() -> Self {
            Self {
                hf_repo: "microsoft/Phi-3-mini-4k-instruct-gguf",
                gguf_file: "Phi-3-mini-4k-instruct-q4.gguf",
                tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct",
                display_name: "Phi-3-Mini-4K Q4 (Candle)",
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Model wrapper
    // ──────────────────────────────────────────────────────────────────────────

    pub struct CandleRunner {
        pub model: ModelWeights,
        pub tokenizer: Tokenizer,
        pub device: Device,
        pub config: CandleModelConfig,
        pub model_load_time_ms: u64,
    }

    impl CandleRunner {
        /// Download (or use cached) model + tokeniser, load into memory.
        /// Returns `Err` with a human-readable message on any failure so the
        /// caller can skip the Candle section gracefully.
        pub fn load(config: CandleModelConfig) -> Result<Self, String> {
            // ── Device selection ────────────────────────────────────────────
            #[cfg(feature = "metal")]
            let device = Device::new_metal(0)
                .map_err(|e| format!("Metal device error: {e}"))?;

            #[cfg(all(feature = "cuda", not(feature = "metal")))]
            let device = Device::new_cuda(0)
                .map_err(|e| format!("CUDA device error: {e}"))?;

            #[cfg(not(any(feature = "metal", feature = "cuda")))]
            let device = Device::Cpu;

            println!(
                "  [Candle] Device: {}",
                match &device {
                    Device::Cpu => "CPU".to_string(),
                    Device::Metal(_) => "Metal (Apple GPU)".to_string(),
                    Device::Cuda(_) => "CUDA".to_string(),
                    _ => "Other".to_string(),
                }
            );

            // ── Download / cache model weights ──────────────────────────────
            println!(
                "  [Candle] Fetching {} …",
                config.gguf_file
            );
            let api = Api::new().map_err(|e| format!("HF Hub API error: {e}"))?;
            let repo = api.repo(Repo::new(config.hf_repo.to_string(), RepoType::Model));
            let gguf_path: PathBuf = repo
                .get(config.gguf_file)
                .map_err(|e| format!("Model download error: {e}"))?;

            // ── Download / cache tokeniser ──────────────────────────────────
            println!("  [Candle] Fetching tokeniser …");
            let tok_repo = api.repo(Repo::new(
                config.tokenizer_repo.to_string(),
                RepoType::Model,
            ));
            let tokenizer_path = tok_repo
                .get("tokenizer.json")
                .map_err(|e| format!("Tokeniser download error: {e}"))?;

            // ── Load model ──────────────────────────────────────────────────
            println!("  [Candle] Loading weights into memory …");
            let t0 = Instant::now();
            let mut file = std::fs::File::open(&gguf_path)
                .map_err(|e| format!("Cannot open GGUF file: {e}"))?;
            let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .map_err(|e| format!("GGUF parse error: {e}"))?;
            let model = ModelWeights::from_gguf(gguf_content, &mut file, &device)
                .map_err(|e| format!("Model load error: {e}"))?;
            let model_load_time_ms = t0.elapsed().as_millis() as u64;
            println!(
                "  [Candle] Model loaded in {model_load_time_ms} ms"
            );

            // ── Load tokeniser ──────────────────────────────────────────────
            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| format!("Tokeniser load error: {e}"))?;

            Ok(Self {
                model,
                tokenizer,
                device,
                config,
                model_load_time_ms,
            })
        }

        // ── Inference ───────────────────────────────────────────────────────

        /// Run greedy generation for a single prompt, returning detailed stats.
        pub fn generate(&mut self, prompt: &BenchmarkPrompt) -> Result<GenerationStats, String> {
            // Tokenise
            let encoding = self
                .tokenizer
                .encode(prompt.text, true)
                .map_err(|e| format!("Tokenise error: {e}"))?;
            let input_ids: Vec<u32> = encoding.get_ids().to_vec();
            let prompt_token_count = input_ids.len();

            // Convert to tensor – shape [1, seq_len]
            let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
                .map_err(|e| format!("Tensor error: {e}"))?
                .unsqueeze(0)
                .map_err(|e| format!("Unsqueeze error: {e}"))?;

            // Greedy logits processor (temperature=0 → argmax)
            let mut logits_processor = LogitsProcessor::new(
                /* seed */ 42,
                /* temperature */ Some(0.0),
                /* top_p */ None,
            );

            let eos_token_id = self
                .tokenizer
                .token_to_id("</s>")
                .or_else(|| self.tokenizer.token_to_id("<|end_of_text|>"))
                .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
                .unwrap_or(2); // llama default EOS

            // ── Prefill ──────────────────────────────────────────────────────
            let prefill_start = Instant::now();
            let logits = self
                .model
                .forward(&input_tensor, 0)
                .map_err(|e| format!("Prefill forward error: {e}"))?;
            // Sample the first new token
            let first_token = Self::sample_token(&logits, &mut logits_processor)?;
            let ttft = prefill_start.elapsed();

            if first_token == eos_token_id {
                return Ok(GenerationStats::new(
                    prompt.name,
                    &self.config.display_name,
                    prompt_token_count,
                    0,
                    ttft,
                    std::time::Duration::ZERO,
                ));
            }

            // ── Decode loop ──────────────────────────────────────────────────
            let mut generated = vec![first_token];
            let mut per_token_times = Vec::with_capacity(prompt.max_new_tokens);

            let mut pos = prompt_token_count;
            let mut current_token = first_token;

            while generated.len() < prompt.max_new_tokens {
                let step_start = Instant::now();
                let token_tensor = Tensor::new(&[current_token], &self.device)
                    .map_err(|e| format!("Token tensor error: {e}"))?
                    .unsqueeze(0)
                    .map_err(|e| format!("Unsqueeze error: {e}"))?;

                let step_logits = self
                    .model
                    .forward(&token_tensor, pos)
                    .map_err(|e| format!("Decode forward error: {e}"))?;

                current_token = Self::sample_token(&step_logits, &mut logits_processor)?;
                per_token_times.push(step_start.elapsed());

                if current_token == eos_token_id {
                    break;
                }
                generated.push(current_token);
                pos += 1;
            }

            let decode_time: std::time::Duration = per_token_times.iter().sum();
            let output_tokens = generated.len();

            let mut stats = GenerationStats::new(
                prompt.name,
                &self.config.display_name,
                prompt_token_count,
                output_tokens,
                ttft,
                decode_time,
            );
            stats.per_token_latencies_us = per_token_times
                .iter()
                .map(|d| d.as_micros() as u64)
                .collect();
            stats.compute_percentiles();

            Ok(stats)
        }

        fn sample_token(
            logits: &Tensor,
            processor: &mut LogitsProcessor,
        ) -> Result<u32, String> {
            // logits shape: [1, vocab_size]  or  [1, seq, vocab]
            let logits = logits.squeeze(0).map_err(|e| e.to_string())?;
            // If 2-D (seq, vocab) grab last position
            let logits = if logits.dims().len() == 2 {
                let seq = logits.dim(0).map_err(|e| e.to_string())?;
                logits
                    .narrow(0, seq - 1, 1)
                    .map_err(|e| e.to_string())?
                    .squeeze(0)
                    .map_err(|e| e.to_string())?
            } else {
                logits
            };
            processor
                .sample(&logits)
                .map_err(|e| format!("Sampling error: {e}"))
        }
    }
}

#[cfg(not(feature = "candle"))]
pub mod inner {
    // Stub so the rest of the crate compiles without the feature.
    pub struct CandleRunner;
}
