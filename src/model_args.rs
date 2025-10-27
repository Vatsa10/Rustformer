use clap::Parser;
use serde::{Deserialize, Serialize};

/// A Rust implementation of the Transformer model from "Attention Is All You Need".
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(author, version, about, long_about = None)]
pub struct ModelArgs {
    // --- Architecture ---
    /// Maximum sequence length.
    #[arg(long, default_value_t = 512)]
    pub block_size: usize,

    /// Embedding dimensions (d_model).
    #[arg(long, default_value_t = 512)]
    pub embeddings_dims: usize,

    /// Number of attention heads.
    #[arg(long, default_value_t = 8)]
    pub no_of_heads: usize,

    /// Number of encoder layers.
    #[arg(long, default_value_t = 6)]
    pub no_of_encoder_layers: usize,

    /// Number of decoder layers.
    #[arg(long, default_value_t = 6)]
    pub no_of_decoder_layers: usize,

    /// Dimension of the feed-forward network.
    #[arg(long, default_value_t = 2048)]
    pub d_ff: usize,

    /// Vocabulary size.
    #[arg(long, default_value_t = 37000)]
    pub vocab_size: usize,

    // --- Optimization ---
    /// Maximum learning rate for the AdamW optimizer.
    #[arg(long, default_value_t = 6e-4)]
    pub max_lr: f64,

    /// Minimum learning rate for the cosine decay schedule.
    #[arg(long, default_value_t = 6e-5)]
    pub min_lr: f64,

    /// Number of warmup steps for the learning rate scheduler.
    #[arg(long, default_value_t = 4000)]
    pub warmup_steps: usize,

    /// Total number of training steps.
    #[arg(long, default_value_t = 100000)]
    pub max_steps: usize,

    /// Weight decay for the optimizer.
    #[arg(long, default_value_t = 0.1)]
    pub weight_decay: f64,

    /// Adam optimizer beta1 parameter.
    #[arg(long, default_value_t = 0.9)]
    pub beta1: f64,

    /// Adam optimizer beta2 parameter.
    #[arg(long, default_value_t = 0.98)]
    pub beta2: f64,

    /// Adam optimizer epsilon parameter.
    #[arg(long, default_value_t = 1e-9)]
    pub epsilon: f64,

    /// Maximum norm for gradient clipping.
    #[arg(long, default_value_t = 1.0)]
    pub grad_clip_max_norm: f64,

    // --- Training ---
    /// Batch size (number of sequences per batch).
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// Dropout probability.
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,

    /// Label smoothing factor.
    #[arg(long, default_value_t = 0.1)]
    pub label_smoothing: f64,

    /// Whether to use mixed-precision training (BF16).
    #[arg(long, default_value_t = true)]
    pub use_mixed_precision: bool,

    // --- Generation ---
    /// Top-k sampling parameter.
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,

    /// Sampling temperature.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,

    /// Repetition penalty during generation.
    #[arg(long, default_value_t = 1.2)]
    pub repetition_penalty: f64,

    /// Beam width for beam search decoding.
    #[arg(long, default_value_t = 4)]
    pub beam_width: usize,

    // --- Monitoring ---
    /// Interval for logging training metrics.
    #[arg(long, default_value_t = 100)]
    pub log_interval: usize,

    /// Interval for running evaluation.
    #[arg(long, default_value_t = 1000)]
    pub eval_interval: usize,

    /// Interval for saving model checkpoints.
    #[arg(long, default_value_t = 5000)]
    pub save_interval: usize,

    /// Interval for generating sample outputs.
    #[arg(long, default_value_t = 500)]
    pub generate_samples_interval: usize,
}

impl ModelArgs {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.embeddings_dims % self.no_of_heads != 0 {
            return Err("embeddings_dims must be divisible by no_of_heads".to_string());
        }

        if self.max_lr <= self.min_lr {
            return Err("max_lr must be greater than min_lr".to_string());
        }

        if self.warmup_steps >= self.max_steps {
            return Err("warmup_steps must be less than max_steps".to_string());
        }

        if self.grad_clip_max_norm <= 0.0 {
            return Err("grad_clip_max_norm must be positive".to_string());
        }

        if self.temperature <= 0.0 {
            return Err("temperature must be positive".to_string());
        }

        if self.top_k == 0 {
            return Err("top_k must be positive".to_string());
        }

        if self.beam_width == 0 {
            return Err("beam_width must be positive".to_string());
        }

        Ok(())
    }

    /// Get current learning rate based on step (warmup + cosine decay)
    pub fn get_learning_rate(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.max_lr * (step as f64) / (self.warmup_steps as f64)
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f64 / (self.max_steps - self.warmup_steps) as f64;
            let progress = progress.min(1.0);
            let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        }
    }
}
