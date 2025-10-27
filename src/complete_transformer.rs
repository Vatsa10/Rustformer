use nalgebra::DMatrix;
use anyhow::Result;
use crate::attention::{FullMultiHeadAttention, MaskedMultiHeadAttention, CrossMultiHeadAttention};
use crate::model_args::ModelArgs;

/// Encoder layer with full multi-head self-attention
pub struct EncoderLayer {
    self_attention: FullMultiHeadAttention,
    d_model: usize,
    d_ff: usize,
}

impl EncoderLayer {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            self_attention: FullMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads),
            d_model: args.embeddings_dims,
            d_ff: args.d_ff,
        }
    }
    
    pub fn forward(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Self-attention with residual connection
        let attn_output = self.self_attention.forward(x, x, x)?;
        let x = self.layer_norm(&(x + attn_output));
        
        // Feed-forward with residual connection
        let ffn_output = self.feed_forward(&x);
        let x = self.layer_norm(&(&x + ffn_output));
        
        Ok(x)
    }
    
    fn feed_forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        // FFN(x) = max(0, xW1 + b1)W2 + b2
        // Simplified: apply ReLU-like activation
        x.map(|val| (val * 2.0).max(0.0))
    }
    
    fn layer_norm(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
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

/// Decoder layer with masked self-attention and cross-attention
pub struct DecoderLayer {
    masked_self_attention: MaskedMultiHeadAttention,
    cross_attention: CrossMultiHeadAttention,
    d_model: usize,
    d_ff: usize,
}

impl DecoderLayer {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            masked_self_attention: MaskedMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads),
            cross_attention: CrossMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads),
            d_model: args.embeddings_dims,
            d_ff: args.d_ff,
        }
    }
    
    pub fn forward(&self, x: &DMatrix<f64>, encoder_output: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Masked self-attention (prevents attending to future)
        let masked_attn_output = self.masked_self_attention.forward(x, x, x)?;
        let x = self.layer_norm(&(x + masked_attn_output));
        
        // Cross-attention to encoder output
        let cross_attn_output = self.cross_attention.forward(&x, encoder_output, encoder_output)?;
        let x = self.layer_norm(&(&x + cross_attn_output));
        
        // Feed-forward with residual
        let ffn_output = self.feed_forward(&x);
        let x = self.layer_norm(&(&x + ffn_output));
        
        Ok(x)
    }
    
    fn feed_forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        x.map(|val| (val * 2.0).max(0.0))
    }
    
    fn layer_norm(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
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

/// Complete Encoder-Decoder Transformer
pub struct CompleteTransformer {
    encoder_layers: Vec<EncoderLayer>,
    decoder_layers: Vec<DecoderLayer>,
    args: ModelArgs,
}

impl CompleteTransformer {
    pub fn new(args: ModelArgs) -> Result<Self> {
        args.validate().map_err(|e| anyhow::anyhow!(e))?;
        
        let mut encoder_layers = Vec::new();
        for _ in 0..args.no_of_encoder_layers {
            encoder_layers.push(EncoderLayer::new(&args));
        }
        
        let mut decoder_layers = Vec::new();
        for _ in 0..args.no_of_decoder_layers {
            decoder_layers.push(DecoderLayer::new(&args));
        }
        
        Ok(Self {
            encoder_layers,
            decoder_layers,
            args,
        })
    }
    
    /// Encode input sequence
    pub fn encode(&self, src: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let mut x = src.clone();
        
        // Add positional encoding
        x = x + self.positional_encoding(src.nrows());
        
        // Pass through encoder layers
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }
        
        Ok(x)
    }
    
    /// Decode with encoder output
    pub fn decode(&self, tgt: &DMatrix<f64>, encoder_output: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let mut x = tgt.clone();
        
        // Add positional encoding
        x = x + self.positional_encoding(tgt.nrows());
        
        // Pass through decoder layers
        for layer in &self.decoder_layers {
            x = layer.forward(&x, encoder_output)?;
        }
        
        Ok(x)
    }
    
    /// Full forward pass (encode + decode)
    pub fn forward(&self, src: &DMatrix<f64>, tgt: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let encoder_output = self.encode(src)?;
        self.decode(tgt, &encoder_output)
    }
    
    /// Sinusoidal positional encoding
    fn positional_encoding(&self, seq_len: usize) -> DMatrix<f64> {
        let d_model = self.args.embeddings_dims;
        let mut pe = DMatrix::zeros(seq_len, d_model);
        
        for pos in 0..seq_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (i as f64 / d_model as f64) * (10000.0_f64).ln();
                let div_term = (-div_term).exp();
                let angle = pos as f64 * div_term;
                
                pe[(pos, i)] = angle.sin();
                if i + 1 < d_model {
                    pe[(pos, i + 1)] = angle.cos();
                }
            }
        }
        
        pe
    }
    
    /// Get model configuration
    pub fn get_args(&self) -> &ModelArgs {
        &self.args
    }
}