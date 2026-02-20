/// Fixed set of benchmark prompts used for every backend so every run
/// processes exactly the same bytes through the same model.
///
/// "Apples-to-apples" rules:
///  • All backends receive the same tokenised IDs (verified by printing token
///    counts before the run).
///  • Generation is greedy (temperature = 0, no sampling randomness).
///  • Only inference time is measured – model loading / warm-up are excluded.
///  • The same `max_new_tokens` budget is given to every backend.

/// A single benchmark scenario: a name, the prompt text, and how many new
/// tokens should be generated.
#[derive(Debug, Clone)]
pub struct BenchmarkPrompt {
    /// Human-readable name shown in results tables.
    pub name: &'static str,
    /// The actual prompt text fed to the model.
    pub text: &'static str,
    /// Number of NEW tokens to generate (output, not input).
    pub max_new_tokens: usize,
}

/// The canonical benchmark prompt suite.
///
/// These are intentionally varied in length and domain to surface differences
/// in prefill latency (long prompts stress the attention mechanism) versus
/// decode throughput (lots of `max_new_tokens` stresses the token-by-token
/// loop).
pub const BENCHMARK_PROMPTS: &[BenchmarkPrompt] = &[
    // ── Short prompts: stresses decode throughput ----------------------------
    BenchmarkPrompt {
        name: "short-creative",
        text: "Write a haiku about Rust programming.",
        max_new_tokens: 64,
    },
    BenchmarkPrompt {
        name: "short-factual",
        text: "Explain what a transformer model is in one paragraph.",
        max_new_tokens: 128,
    },
    // ── Medium prompts: balanced prefill + decode ----------------------------
    BenchmarkPrompt {
        name: "medium-code",
        text: "Write a Rust function that computes the Fibonacci sequence \
               iteratively and returns the nth number. Include error handling \
               for negative inputs.",
        max_new_tokens: 256,
    },
    BenchmarkPrompt {
        name: "medium-reasoning",
        text: "A train leaves city A at 60 mph. Another train leaves city B \
               (200 miles away) at 80 mph heading toward city A. When and \
               where do they meet? Show your work step by step.",
        max_new_tokens: 200,
    },
    // ── Long prompt: stresses prefill (attention over many input tokens) -----
    BenchmarkPrompt {
        name: "long-summarise",
        text: "Summarise the following passage in three bullet points:\n\
               \n\
               The Rust programming language was originally designed by Graydon \
               Hoare at Mozilla Research in 2006. It was first released publicly \
               in 2010 and reached version 1.0 in May 2015. Rust is designed to \
               be a systems programming language with a focus on three goals: \
               safety, speed, and concurrency. It achieves memory safety without \
               a garbage collector, instead using a system of ownership with rules \
               that the compiler checks at compile time. The language has grown \
               significantly in popularity, being voted the 'most loved \
               programming language' in the Stack Overflow Developer Survey for \
               nine consecutive years from 2016 through 2024. It is now used in \
               the Linux kernel, Android, Windows, and major web infrastructure \
               projects worldwide.",
        max_new_tokens: 150,
    },
];

/// A quick single-prompt variant used for smoke tests and warmup.
pub const WARMUP_PROMPT: BenchmarkPrompt = BenchmarkPrompt {
    name: "warmup",
    text: "Hello",
    max_new_tokens: 16,
};
