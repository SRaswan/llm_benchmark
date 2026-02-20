use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

/// Configuration for the GPT model
#[derive(Config, Debug)]
pub struct GptConfig {
    /// Vocabulary size
    #[config(default = 1000)]
    pub vocab_size: usize,
    
    /// Hidden dimension
    #[config(default = 256)]
    pub hidden_size: usize,
    
    /// Number of transformer layers
    #[config(default = 4)]
    pub num_layers: usize,
    
    /// Number of attention heads
    #[config(default = 4)]
    pub num_heads: usize,
    
    /// Maximum sequence length
    #[config(default = 128)]
    pub max_seq_len: usize,
    
    /// Dropout probability
    #[config(default = 0.1)]
    pub dropout: f64,
    
    /// Feed-forward intermediate size
    #[config(default = 1024)]
    pub intermediate_size: usize,
}

impl GptConfig {
    /// Create a very small config for fast benchmarking
    pub fn tiny() -> Self {
        Self::new()
            .with_vocab_size(512)
            .with_hidden_size(128)
            .with_num_layers(2)
            .with_num_heads(2)
            .with_max_seq_len(64)
            .with_intermediate_size(256)
    }

    /// Create a small config
    pub fn small() -> Self {
        Self::new()
            .with_vocab_size(2048)
            .with_hidden_size(256)
            .with_num_layers(4)
            .with_num_heads(4)
            .with_max_seq_len(128)
            .with_intermediate_size(512)
    }
}

/// GPT-style transformer model
#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    ln_final: LayerNorm<B>,
    lm_head: Linear<B>,
    dropout: Dropout,
    max_seq_len: usize,
}

impl<B: Backend> Gpt<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.hidden_size)
            .init(device);
        
        let position_embedding = EmbeddingConfig::new(config.max_seq_len, config.hidden_size)
            .init(device);
        
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(TransformerBlock::new(config, device));
        }
        
        let ln_final = LayerNormConfig::new(config.hidden_size).init(device);
        
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);
        
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self {
            token_embedding,
            position_embedding,
            layers,
            ln_final,
            lm_head,
            dropout,
            max_seq_len: config.max_seq_len,
        }
    }

    /// Forward pass through the model
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        assert!(seq_len <= self.max_seq_len, "Sequence length exceeds maximum");
        
        // Create position indices [0, 1, 2, ..., seq_len-1]
        let positions = Tensor::arange(0..seq_len as i64, &input_ids.device())
            .reshape([1, seq_len])
            .repeat(0.0, batch_size);
        
        // Token + position embeddings
        let token_emb = self.token_embedding.forward(input_ids);
        let pos_emb = self.position_embedding.forward(positions);
        
        let mut hidden = token_emb + pos_emb;
        hidden = self.dropout.forward(hidden);
        
        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        
        // Final layer norm and LM head
        hidden = self.ln_final.forward(hidden);
        self.lm_head.forward(hidden)
    }

    /// Generate logits for next token prediction
    pub fn generate_logits(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let logits = self.forward(input_ids);
        
        // Get the logits for the last token in the sequence
        let [_batch_size, seq_len, _vocab_size] = logits.dims();
        logits.slice([0..1, (seq_len - 1)..seq_len]).squeeze(1)
    }
}

/// Single transformer block
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    ln1: LayerNorm<B>,
    mlp: Mlp<B>,
    ln2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(config.hidden_size, config.num_heads)
            .with_dropout(config.dropout)
            .init(device);
        
        let ln1 = LayerNormConfig::new(config.hidden_size).init(device);
        let ln2 = LayerNormConfig::new(config.hidden_size).init(device);
        let mlp = Mlp::new(config, device);
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self {
            attention,
            ln1,
            mlp,
            ln2,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm architecture
        // Self-attention with residual
        let normed = self.ln1.forward(x.clone());
        let attn_input = MhaInput::self_attn(normed);
        let attn_out = self.attention.forward(attn_input);
        let x = x + self.dropout.forward(attn_out.context);
        
        // MLP with residual
        let normed = self.ln2.forward(x.clone());
        let mlp_out = self.mlp.forward(normed);
        x + self.dropout.forward(mlp_out)
    }
}

/// Multi-layer perceptron (feed-forward network)
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(config.hidden_size, config.intermediate_size).init(device);
        let fc2 = LinearConfig::new(config.intermediate_size, config.hidden_size).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        
        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}
