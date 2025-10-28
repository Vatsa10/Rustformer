use serde::{Serialize, Deserialize};

/// Comprehensive model configuration matching the requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArgs {
    // Architecture
    pub block_size: usize,
    pub batch_size: usize,
    pub embeddings_dims: usize,
    pub no_of_heads: usize,
    pub no_of_encoder_layers: usize,
    pub no_of_decoder_layers: usize,
    pub vocab_size: usize,
    pub d_ff: usize,
    
    // Optimization
    pub max_lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub grad_clip_max_norm: f64,
    
    // Training
    pub dropout: f64,
    pub label_smoothing: f64,
    pub use_mixed_precision: bool,
    
    // Generation
    pub top_k: usize,
    pub temperature: f64,
    pub repetition_penalty: f64,
    pub beam_width: usize,
    
    // Monitoring
    pub log_interval: usize,
    pub eval_interval: usize,
    pub save_interval: usize,
    pub generate_samples_interval: usize,
}

impl Default for ModelArgs {
    fn default() -> Self {
        Self {
            // Architecture (matching paper defaults)
            block_size: 512,
            batch_size: 32,
            embeddings_dims: 512,
            no_of_heads: 8,
            no_of_encoder_layers: 6,
            no_of_decoder_layers: 6,
            vocab_size: 37000,
            d_ff: 2048,
            
            // Optimization (AdamW with warmup + cosine decay)
            max_lr: 6e-4,
            min_lr: 6e-5,
            warmup_steps: 4000,
            max_steps: 100000,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.98,
            epsilon: 1e-9,
            grad_clip_max_norm: 1.0,
            
            // Training
            dropout: 0.1,
            label_smoothing: 0.1,
            use_mixed_precision: true,
            
            // Generation
            top_k: 50,
            temperature: 1.0,
            repetition_penalty: 1.2,
            beam_width: 4,
            
            // Monitoring
            log_interval: 100,
            eval_interval: 1000,
            save_interval: 5000,
            generate_samples_interval: 500,
        }
    }
}

impl ModelArgs {
    pub fn new() -> Self {
        Self::default()
    }
    
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
    
    /// Get current learning rate based on step
    /// Uses the exact formula from "Attention Is All You Need" paper:
    /// lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
    pub fn get_learning_rate(&self, step: usize) -> f64 {
        let step = (step + 1) as f64; // Add 1 to avoid division by zero
        let d_model = self.embeddings_dims as f64;
        let warmup = self.warmup_steps as f64;
        
        let scale = d_model.powf(-0.5);
        let lr_factor = step.powf(-0.5).min(step * warmup.powf(-1.5));
        
        scale * lr_factor
    }
}