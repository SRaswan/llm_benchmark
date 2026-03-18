/// Pure-Rust LLM inference using HuggingFace Candle.
///
/// This module loads a GGUF-quantised model from the HuggingFace Hub and runs
/// autoregressive generation, measuring the same metrics as the Burn benchmarks
/// so the results can be directly compared.
///
/// Notes:
/// - The current runtime loader is `quantized_llama::ModelWeights`, so only
///   LLaMA-loader-compatible GGUFs should be marked as supported.
/// - TinyLlama works with this path.
/// - Phi/Qwen entries can exist in the registry, but should be marked unsupported
///   until you add dedicated loaders.

#[cfg(feature = "candle")]
pub mod inner {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use candle_core::{Device, Tensor};
    use candle_transformers::generation::LogitsProcessor;
    use candle_transformers::models::quantized_llama::ModelWeights;
    use hf_hub::{
        api::sync::{Api, ApiBuilder},
        Repo, RepoType,
    };
    use tokenizers::Tokenizer;

    use crate::benchmark::{GenerationStats, MemoryStats};
    use crate::prompts::BenchmarkPrompt;

    // -------------------------------------------------------------------------
    // Model registry
    // -------------------------------------------------------------------------

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum CandleModelKind {
        TinyLlama,
        Llama32_1B,
        Phi4,
        Qwen25_05B,
    }

    impl CandleModelKind {
        pub fn parse(s: &str) -> Option<Self> {
            match s.to_lowercase().as_str() {
                "tinyllama" | "tiny_llama" => Some(Self::TinyLlama),
                "llama32" | "llama3.2" | "llama3.2-1b" | "llama-3.2-1b" => Some(Self::Llama32_1B),
                "phi4" | "phi-4" => Some(Self::Phi4),
                "qwen" | "qwen2.5" | "qwen2.5-0.5b" | "qwen2_5_0_5b" => Some(Self::Qwen25_05B),
                _ => None,
            }
        }

        pub fn all() -> Vec<Self> {
            vec![
                Self::TinyLlama,
                Self::Llama32_1B,
                Self::Phi4,
                Self::Qwen25_05B,
            ]
        }

        pub fn config(&self) -> CandleModelConfig {
            match self {
                Self::TinyLlama => CandleModelConfig::tiny_llama(),
                Self::Llama32_1B => CandleModelConfig::llama32_1b(),
                Self::Phi4 => CandleModelConfig::phi4(),
                Self::Qwen25_05B => CandleModelConfig::qwen25_05b(),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct CandleModelConfig {
        pub kind: CandleModelKind,
        pub hf_repo: &'static str,
        pub gguf_file: &'static str,
        pub tokenizer_repo: &'static str,
        pub display_name: &'static str,
        pub requires_token: bool,
        pub supported_by_current_loader: bool,
        pub local_dir_name: &'static str,
    }

    impl CandleModelConfig {
        pub fn tiny_llama() -> Self {
            Self {
                kind: CandleModelKind::TinyLlama,
                hf_repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                gguf_file: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                tokenizer_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                display_name: "TinyLlama-1.1B Q4_K_M (Candle)",
                requires_token: false,
                supported_by_current_loader: true,
                local_dir_name: "tinyllama",
            }
        }

        pub fn llama32_1b() -> Self {
            Self {
                kind: CandleModelKind::Llama32_1B,
                hf_repo: "unsloth/Llama-3.2-1B-Instruct-GGUF",
                gguf_file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                tokenizer_repo: "meta-llama/Llama-3.2-1B-Instruct",
                display_name: "Llama-3.2-1B-Instruct Q4_K_M (Candle)",
                requires_token: true,
                supported_by_current_loader: true,
                local_dir_name: "llama32",
            }
        }

        pub fn phi4() -> Self {
            Self {
                kind: CandleModelKind::Phi4,
                hf_repo: "microsoft/phi-4",
                gguf_file: "phi-4-q4.gguf",
                tokenizer_repo: "microsoft/phi-4",
                display_name: "Phi-4 (placeholder until dedicated Candle loader exists)",
                requires_token: true,
                supported_by_current_loader: false,
                local_dir_name: "phi4",
            }
        }

        pub fn qwen25_05b() -> Self {
            Self {
                kind: CandleModelKind::Qwen25_05B,
                hf_repo: "Qwen/Qwen2.5-0.5B-Instruct",
                gguf_file: "qwen2.5-0.5b-instruct-q4.gguf",
                tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct",
                display_name: "Qwen2.5-0.5B (placeholder until dedicated Candle loader exists)",
                requires_token: false,
                supported_by_current_loader: false,
                local_dir_name: "qwen",
            }
        }
    }

    // -------------------------------------------------------------------------
    // Runner
    // -------------------------------------------------------------------------

    pub struct CandleRunner {
        pub model: ModelWeights,
        pub tokenizer: Tokenizer,
        pub device: Device,
        pub config: CandleModelConfig,
        pub model_load_time_ms: u64,
    }

    impl CandleRunner {
        pub fn load(config: CandleModelConfig, hf_token: Option<&str>) -> Result<Self, String> {
            if !config.supported_by_current_loader {
                return Err(format!(
                    "{} is not supported by the current Candle loader yet. \
                     This runner currently uses quantized_llama::ModelWeights. \
                     Add a dedicated loader for this architecture first.",
                    config.display_name
                ));
            }

            let api = build_hf_api(hf_token)?;

            #[cfg(feature = "metal")]
            let device = Device::new_metal(0).map_err(|e| format!("Metal device error: {e}"))?;

            #[cfg(all(feature = "cuda", not(feature = "metal")))]
            let device = Device::new_cuda(0).map_err(|e| format!("CUDA device error: {e}"))?;

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

            let (gguf_path, tokenizer_path) = resolve_model_paths(&api, &config)?;

            println!("  [Candle] Loading weights into memory...");
            let t0 = Instant::now();

            let mut file = std::fs::File::open(&gguf_path)
                .map_err(|e| format!("Cannot open GGUF file {}: {e}", gguf_path.display()))?;
            let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)
                .map_err(|e| format!("GGUF parse error: {e}"))?;
            let model = ModelWeights::from_gguf(gguf_content, &mut file, &device)
                .map_err(|e| format!("Model load error: {e}"))?;

            let model_load_time_ms = t0.elapsed().as_millis() as u64;
            println!("  [Candle] Model loaded in {model_load_time_ms} ms");

            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| format!("Tokenizer load error: {e}"))?;

            Ok(Self {
                model,
                tokenizer,
                device,
                config,
                model_load_time_ms,
            })
        }

        pub fn generate(&mut self, prompt: &BenchmarkPrompt) -> Result<GenerationStats, String> {
            let encoding = self
                .tokenizer
                .encode(prompt.text, true)
                .map_err(|e| format!("Tokenize error: {e}"))?;
            let input_ids: Vec<u32> = encoding.get_ids().to_vec();
            let prompt_token_count = input_ids.len();

            let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
                .map_err(|e| format!("Tensor error: {e}"))?
                .unsqueeze(0)
                .map_err(|e| format!("Unsqueeze error: {e}"))?;

            let mut logits_processor = LogitsProcessor::new(42, Some(0.0), None);

            let eos_token_id = self
                .tokenizer
                .token_to_id("</s>")
                .or_else(|| self.tokenizer.token_to_id("<|end_of_text|>"))
                .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
                .unwrap_or(2);

            let prefill_start = Instant::now();
            let logits = self
                .model
                .forward(&input_tensor, 0)
                .map_err(|e| format!("Prefill forward error: {e}"))?;
            let first_token = Self::sample_token(&logits, &mut logits_processor)?;
            let ttft = prefill_start.elapsed();

            if first_token == eos_token_id {
                return Ok(GenerationStats::new(
                    prompt.name,
                    &self.config.display_name,
                    prompt_token_count,
                    0,
                    ttft,
                    Duration::ZERO,
                    MemoryStats {
                        rss_bytes: 0,
                        vsz_bytes: 0,
                    },
                ));
            }

            let memory_before = MemoryStats::current().unwrap_or_else(|_| MemoryStats {
                rss_bytes: 0,
                vsz_bytes: 0,
            });

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

            let memory_after = MemoryStats::current().unwrap_or_else(|_| MemoryStats {
                rss_bytes: 0,
                vsz_bytes: 0,
            });

            let decode_time: Duration = per_token_times.iter().copied().sum();
            let output_tokens = generated.len();

            let peak_memory = MemoryStats {
                rss_bytes: memory_after
                    .rss_bytes
                    .saturating_sub(memory_before.rss_bytes),
                vsz_bytes: memory_after
                    .vsz_bytes
                    .saturating_sub(memory_before.vsz_bytes),
            };

            let mut stats = GenerationStats::new(
                prompt.name,
                &self.config.display_name,
                prompt_token_count,
                output_tokens,
                ttft,
                decode_time,
                peak_memory,
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
            let logits = logits.squeeze(0).map_err(|e| e.to_string())?;

            let logits = if logits.dims().len() == 2 {
                let seq = logits.dim(0).map_err(|e| e.to_string())?;
                logits
                    .narrow(0, seq.saturating_sub(1), 1)
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

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn resolve_model_paths(
        api: &Api,
        config: &CandleModelConfig,
    ) -> Result<(PathBuf, PathBuf), String> {
        let gguf_env = std::env::var("CANDLE_GGUF_PATH").ok();
        let tok_env = std::env::var("CANDLE_TOKENIZER_PATH").ok();

        if gguf_env.is_some() || tok_env.is_some() {
            let gguf_path = gguf_env
                .ok_or_else(|| {
                    "CANDLE_GGUF_PATH is required when using local override paths".to_string()
                })
                .map(PathBuf::from)?;
            let tokenizer_path = tok_env
                .ok_or_else(|| {
                    "CANDLE_TOKENIZER_PATH is required when using local override paths"
                        .to_string()
                })
                .map(PathBuf::from)?;
            return Ok((gguf_path, tokenizer_path));
        }

        if let Some((gguf_path, tokenizer_path)) = try_local_model_folder(config)? {
            return Ok((gguf_path, tokenizer_path));
        }

        println!("  [Candle] Fetching {} ...", config.gguf_file);
        let repo = api.repo(Repo::new(config.hf_repo.to_string(), RepoType::Model));
        let gguf_path = repo
            .get(config.gguf_file)
            .map_err(|e| format!("Model download error: {e}"))?;

        println!("  [Candle] Fetching tokenizer ...");
        let tok_repo = api.repo(Repo::new(
            config.tokenizer_repo.to_string(),
            RepoType::Model,
        ));
        let tokenizer_path = tok_repo.get("tokenizer.json").map_err(|e| {
            let mut msg = format!("Tokenizer download error: {e}");
            if msg.contains("RelativeUrlWithoutBase") {
                msg.push_str(
                    " (this often means the repo is gated; set HF_TOKEN / HUGGING_FACE_HUB_TOKEN)",
                );
            }
            msg
        })?;

        Ok((gguf_path, tokenizer_path))
    }

    fn try_local_model_folder(
        config: &CandleModelConfig,
    ) -> Result<Option<(PathBuf, PathBuf)>, String> {
        let folder = PathBuf::from(config.local_dir_name);
        if !folder.is_dir() {
            return Ok(None);
        }

        let tokenizer_path = folder.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(format!(
                "Local folder '{}' found but tokenizer.json is missing",
                config.local_dir_name
            ));
        }

        let preferred = folder.join(config.gguf_file);
        let gguf_path = if preferred.is_file() {
            preferred
        } else {
            let mut ggufs = Vec::new();
            for entry in std::fs::read_dir(&folder)
                .map_err(|e| format!("Cannot read local folder '{}': {e}", config.local_dir_name))?
            {
                let entry = entry
                    .map_err(|e| format!("Cannot read local folder '{}': {e}", config.local_dir_name))?;
                let path = entry.path();
                if path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(false)
                {
                    ggufs.push(path);
                }
            }

            match ggufs.len() {
                0 => {
                    return Err(format!(
                        "Local folder '{}' found but no .gguf file is present",
                        config.local_dir_name
                    ))
                }
                1 => ggufs.remove(0),
                _ => {
                    return Err(format!(
                        "Multiple .gguf files found in '{}'; set CANDLE_GGUF_PATH to pick one",
                        config.local_dir_name
                    ))
                }
            }
        };

        println!("  [Candle] Using local folder: {}", config.local_dir_name);
        Ok(Some((gguf_path, tokenizer_path)))
    }

    fn build_hf_api(explicit_token: Option<&str>) -> Result<Api, String> {
        let token = explicit_token
            .map(|s| s.to_string())
            .or_else(|| std::env::var("LLM_BENCH_HF_TOKEN").ok())
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
            .or_else(|| std::env::var("HF_TOKEN").ok());

        let api = if let Some(token) = token {
            ApiBuilder::new().with_token(Some(token)).build()
        } else {
            Api::new()
        };

        api.map_err(|e| format!("HF Hub API error: {e}"))
    }
}

#[cfg(not(feature = "candle"))]
pub mod inner {
    pub struct CandleRunner;
}