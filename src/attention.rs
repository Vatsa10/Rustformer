use nalgebra::DMatrix;
use anyhow::Result;

/// Create causal mask for masked self-attention in decoder
/// Prevents attending to future positions
pub fn create_causal_mask(seq_len: usize) -> DMatrix<f64> {
    let mut mask = DMatrix::zeros(seq_len, seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[(i, j)] = f64::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Masked Multi-Head Attention for decoder self-attention
/// Prevents attending to future tokens (autoregressive)
pub struct MaskedMultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_k: usize,
}

impl MaskedMultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        Self {
            d_model,
            num_heads,
            d_k,
        }
    }
    
    /// Forward pass with causal masking
    pub fn forward(&self, query: &DMatrix<f64>, key: &DMatrix<f64>, value: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let seq_len = query.nrows();
        let scale = 1.0 / (self.d_k as f64).sqrt();
        
        // Compute attention scores
        let mut scores = query * key.transpose() * scale;
        
        // Apply causal mask
        let mask = create_causal_mask(seq_len);
        scores += mask;
        
        // Apply softmax
        let attention_weights = self.softmax(&scores);
        
        // Apply to values
        Ok(&attention_weights * value)
    }
    
    fn softmax(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = x.clone();
        for mut row in result.row_iter_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            let sum: f64 = row.iter().sum();
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
        result
    }
}

/// Cross Multi-Head Attention for encoder-decoder attention
/// Query from decoder, Key and Value from encoder
pub struct CrossMultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_k: usize,
}

impl CrossMultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        Self {
            d_model,
            num_heads,
            d_k,
        }
    }
    
    /// Forward pass: Q from decoder, K and V from encoder
    pub fn forward(&self, query: &DMatrix<f64>, key: &DMatrix<f64>, value: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let scale = 1.0 / (self.d_k as f64).sqrt();
        
        // Compute attention scores (no masking for cross-attention)
        let scores = query * key.transpose() * scale;
        
        // Apply softmax
        let attention_weights = self.softmax(&scores);
        
        // Apply to values
        Ok(&attention_weights * value)
    }
    
    fn softmax(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = x.clone();
        for mut row in result.row_iter_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            let sum: f64 = row.iter().sum();
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
        result
    }
}

/// Full Multi-Head Attention for encoder self-attention
/// No masking - can attend to all positions
pub struct FullMultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_k: usize,
}

impl FullMultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        Self {
            d_model,
            num_heads,
            d_k,
        }
    }
    
    /// Forward pass without masking
    pub fn forward(&self, query: &DMatrix<f64>, key: &DMatrix<f64>, value: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let scale = 1.0 / (self.d_k as f64).sqrt();
        
        // Compute attention scores
        let scores = query * key.transpose() * scale;
        
        // Apply softmax
        let attention_weights = self.softmax(&scores);
        
        // Apply to values
        Ok(&attention_weights * value)
    }
    
    fn softmax(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = x.clone();
        for mut row in result.row_iter_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            let sum: f64 = row.iter().sum();
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
        result
    }
}