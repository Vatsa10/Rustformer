/// Configuration for the Transformer model based on "Attention Is All You Need" paper
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Model dimension (d_model in paper) - 512 in base model
    pub d_model: usize,
    /// Number of attention heads (h in paper) - 8 in base model  
    pub num_heads: usize,
    /// Dimension of each attention head (d_k = d_v = d_model/h) - 64 in base model
    pub d_k: usize,
    /// Dimension of feed-forward network (d_ff in paper) - 2048 in base model
    pub d_ff: usize,
    /// Number of encoder layers (N in paper) - 6 in base model
    pub num_encoder_layers: usize,
    /// Number of decoder layers (N in paper) - 6 in base model  
    pub num_decoder_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Label smoothing parameter
    pub label_smoothing: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            d_k: 64, // d_model / num_heads
            d_ff: 2048,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            vocab_size: 37000, // As mentioned in paper for WMT 2014 En-De
            max_seq_len: 512,
            dropout: 0.1,
            label_smoothing: 0.1,
        }
    }
}

impl TransformerConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.num_heads != 0 {
            return Err("d_model must be divisible by num_heads".to_string());
        }
        
        if self.d_k != self.d_model / self.num_heads {
            return Err("d_k must equal d_model / num_heads".to_string());
        }
        
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err("dropout must be between 0.0 and 1.0".to_string());
        }
        
        Ok(())
    }
}