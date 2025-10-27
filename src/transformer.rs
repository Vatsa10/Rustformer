use nalgebra::DMatrix;
use crate::config::TransformerConfig;
use anyhow::Result;

/// Scaled Dot-Product Attention implementation
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
pub struct ScaledDotProductAttention {
    d_k: usize,
}

impl ScaledDotProductAttention {
    pub fn new(d_k: usize) -> Self {
        Self { d_k }
    }
    
    /// Apply scaled dot-product attention
    /// q, k, v: matrices of shape (seq_len, d_k)
    pub fn forward(&self, q: &DMatrix<f64>, k: &DMatrix<f64>, v: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let scale = 1.0 / (self.d_k as f64).sqrt();
        
        // Compute attention scores: QK^T / sqrt(d_k)
        let scores = q * k.transpose() * scale;
        
        // Apply softmax row-wise
        let attention_weights = self.softmax(&scores);
        
        // Apply attention weights to values
        let output = &attention_weights * v;
        
        Ok(output)
    }
    
    fn softmax(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = x.clone();
        for mut row in result.row_iter_mut() {
            // Find max for numerical stability
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            // Subtract max and exponentiate
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            
            // Normalize
            let sum: f64 = row.iter().sum();
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
        result
    }
}

/// Multi-Head Attention implementation
pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    attention: ScaledDotProductAttention,
    // In a full implementation, these would be actual weight matrices
    // For this demo, we'll simulate them
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        let attention = ScaledDotProductAttention::new(d_k);
        
        Self {
            num_heads,
            d_model,
            d_k,
            attention,
        }
    }
    
    pub fn forward(&self, query: &DMatrix<f64>, key: &DMatrix<f64>, value: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let _seq_len = query.nrows();
        
        // In a full implementation, we would:
        // 1. Apply linear projections W_Q, W_K, W_V
        // 2. Split into multiple heads
        // 3. Apply attention for each head
        // 4. Concatenate heads
        // 5. Apply output projection W_O
        
        // For this demo, we'll just apply single-head attention
        self.attention.forward(query, key, value)
    }
}

/// Position-wise Feed-Forward Network
/// FFN(x) = max(0, xW1 + b1)W2 + b2
pub struct PositionwiseFeedForward {
    d_model: usize,
    d_ff: usize,
}

impl PositionwiseFeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self { d_model, d_ff }
    }
    
    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        // In a full implementation, this would use actual weight matrices
        // For demo purposes, we'll apply a simple transformation
        x.map(|val| (val * 2.0).max(0.0)) // Simplified ReLU-like activation
    }
}

/// Layer Normalization
pub struct LayerNorm;

impl LayerNorm {
    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = x.clone();
        let eps = 1e-6;
        
        for mut row in result.row_iter_mut() {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let variance = row.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / row.len() as f64;
            let std = (variance + eps).sqrt();
            
            for val in row.iter_mut() {
                *val = (*val - mean) / std;
            }
        }
        result
    }
}

/// Encoder Layer
pub struct EncoderLayer {
    self_attention: MultiHeadAttention,
    ffn: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(d_model, num_heads),
            ffn: PositionwiseFeedForward::new(d_model, d_ff),
            norm1: LayerNorm,
            norm2: LayerNorm,
        }
    }
    
    pub fn forward(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Self-attention with residual connection and layer norm
        let attn_output = self.self_attention.forward(x, x, x)?;
        let x = self.norm1.forward(&(x + attn_output));
        
        // Feed-forward with residual connection and layer norm
        let ffn_output = self.ffn.forward(&x);
        let x = self.norm2.forward(&(&x + ffn_output));
        
        Ok(x)
    }
}

/// Positional Encoding using sinusoidal functions
pub struct PositionalEncoding {
    encoding: DMatrix<f64>,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut encoding = DMatrix::zeros(max_seq_len, d_model);
        
        for pos in 0..max_seq_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (i as f64 / d_model as f64) * (10000.0_f64).ln();
                let div_term = (-div_term).exp();
                let angle = pos as f64 * div_term;
                
                encoding[(pos, i)] = angle.sin();
                if i + 1 < d_model {
                    encoding[(pos, i + 1)] = angle.cos();
                }
            }
        }
        
        Self { encoding }
    }
    
    pub fn forward(&self, seq_len: usize) -> DMatrix<f64> {
        self.encoding.rows(0, seq_len).into()
    }
}

/// Complete Transformer model
pub struct Transformer {
    config: TransformerConfig,
    encoder_layers: Vec<EncoderLayer>,
    positional_encoding: PositionalEncoding,
}

impl Transformer {
    pub fn new(config: &TransformerConfig) -> Result<Self> {
        config.validate().map_err(|e| anyhow::anyhow!(e))?;
        
        let mut encoder_layers = Vec::new();
        for _ in 0..config.num_encoder_layers {
            encoder_layers.push(EncoderLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
            ));
        }
        
        let positional_encoding = PositionalEncoding::new(config.max_seq_len, config.d_model);
        
        Ok(Self {
            config: config.clone(),
            encoder_layers,
            positional_encoding,
        })
    }
    
    /// Forward pass through the transformer
    pub fn forward(&self, input_embeddings: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let seq_len = input_embeddings.nrows();
        
        // Add positional encoding
        let pos_encoding = self.positional_encoding.forward(seq_len);
        let mut x = input_embeddings + pos_encoding;
        
        // Pass through encoder layers
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }
        
        Ok(x)
    }
    
    /// Get model configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transformer_creation() -> Result<()> {
        let config = TransformerConfig::default();
        let transformer = Transformer::new(&config)?;
        
        // Test with dummy input
        let seq_len = 10;
        let input = DMatrix::from_element(seq_len, config.d_model, 0.1);
        
        let output = transformer.forward(&input)?;
        assert_eq!(output.nrows(), seq_len);
        assert_eq!(output.ncols(), config.d_model);
        
        Ok(())
    }
    
    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(100, 512);
        let encoding = pe.forward(10);
        assert_eq!(encoding.nrows(), 10);
        assert_eq!(encoding.ncols(), 512);
    }
}