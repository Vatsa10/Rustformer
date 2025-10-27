use nalgebra::DMatrix;
use anyhow::Result;
use crate::attention::{FullMultiHeadAttention, MaskedMultiHeadAttention, CrossMultiHeadAttention};
use crate::model_args::ModelArgs;

// --- Layer Normalization ---
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

// --- Position-wise Feed-Forward Network ---
pub struct PositionwiseFeedForward {
    d_model: usize,
    d_ff: usize,
}

impl PositionwiseFeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self { d_model, d_ff }
    }
    
    // Simplified FFN with ReLU
    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        // In a real implementation, this would involve two linear layers
        x.map(|val| (val * 2.0).max(0.0))
    }
}

// --- Positional Encoding ---
pub struct PositionalEncoding {
    encoding: DMatrix<f64>,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut encoding = DMatrix::zeros(max_seq_len, d_model);
        
        for pos in 0..max_seq_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (-(i as f64 / d_model as f64) * (10000.0_f64).ln()).exp();
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

// --- Encoder Layer ---
pub struct EncoderLayer {
    self_attention: FullMultiHeadAttention,
    ffn: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            self_attention: FullMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads),
            ffn: PositionwiseFeedForward::new(args.embeddings_dims, args.d_ff),
            norm1: LayerNorm,
            norm2: LayerNorm,
        }
    }

    pub fn forward(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Self-attention with residual and norm
        let attn_output = self.self_attention.forward(x, x, x)?;
        let x = self.norm1.forward(&(x + attn_output));
        
        // FFN with residual and norm
        let ffn_output = self.ffn.forward(&x);
        let x = self.norm2.forward(&(&x + ffn_output));
        
        Ok(x)
    }
}

// --- Encoder ---
pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(args: &ModelArgs) -> Self {
        let mut layers = Vec::new();
        for _ in 0..args.no_of_encoder_layers {
            layers.push(EncoderLayer::new(args));
        }
        Self { layers }
    }

    pub fn forward(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let mut x_out = x.clone();
        for layer in &self.layers {
            x_out = layer.forward(&x_out)?;
        }
        Ok(x_out)
    }
}

// --- Decoder Layer ---
pub struct DecoderLayer {
    masked_self_attention: MaskedMultiHeadAttention,
    cross_attention: CrossMultiHeadAttention,
    ffn: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl DecoderLayer {
    pub fn new(args: &ModelArgs) -> Self {
        Self {
            masked_self_attention: MaskedMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads),
            cross_attention: CrossMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads),
            ffn: PositionwiseFeedForward::new(args.embeddings_dims, args.d_ff),
            norm1: LayerNorm,
            norm2: LayerNorm,
            norm3: LayerNorm,
        }
    }

    pub fn forward(&self, x: &DMatrix<f64>, encoder_output: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Masked self-attention
        let masked_attn_output = self.masked_self_attention.forward(x, x, x)?;
        let x = self.norm1.forward(&(x + masked_attn_output));
        
        // Cross-attention
        let cross_attn_output = self.cross_attention.forward(&x, encoder_output, encoder_output)?;
        let x = self.norm2.forward(&(&x + cross_attn_output));
        
        // Feed-forward
        let ffn_output = self.ffn.forward(&x);
        let x = self.norm3.forward(&(&x + ffn_output));
        
        Ok(x)
    }
}

// --- Decoder ---
pub struct Decoder {
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    pub fn new(args: &ModelArgs) -> Self {
        let mut layers = Vec::new();
        for _ in 0..args.no_of_decoder_layers {
            layers.push(DecoderLayer::new(args));
        }
        Self { layers }
    }

    pub fn forward(&self, x: &DMatrix<f64>, encoder_output: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let mut x_out = x.clone();
        for layer in &self.layers {
            x_out = layer.forward(&x_out, encoder_output)?;
        }
        Ok(x_out)
    }
}

// --- Final Output Layer ---
pub struct OutputLayer {
    vocab_size: usize,
    d_model: usize,
}

impl OutputLayer {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self { vocab_size, d_model }
    }

    // Projects to vocab size and applies softmax
    pub fn forward(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        // Simplified projection: just take a slice of the output
        let projected = x.columns(0, self.vocab_size.min(self.d_model));
        self.softmax(&DMatrix::from(projected))
    }

    fn softmax(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let mut result = x.clone();
        for mut row in result.row_iter_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            row.iter_mut().for_each(|v| *v = (*v - max_val).exp());
            let sum: f64 = row.iter().sum();
            row.iter_mut().for_each(|v| *v /= sum);
        }
        result
    }
}


// --- Complete Transformer ---
pub struct Transformer {
    encoder: Encoder,
    decoder: Decoder,
    positional_encoding: PositionalEncoding,
    output_layer: OutputLayer,
    args: ModelArgs,
}

impl Transformer {
    pub fn new(args: ModelArgs) -> Result<Self> {
        args.validate().map_err(|e| anyhow::anyhow!(e))?;
        
        Ok(Self {
            encoder: Encoder::new(&args),
            decoder: Decoder::new(&args),
            positional_encoding: PositionalEncoding::new(args.block_size, args.embeddings_dims),
            output_layer: OutputLayer::new(args.vocab_size, args.embeddings_dims),
            args,
        })
    }
    
    pub fn encode(&self, src: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let pos_encoding = self.positional_encoding.forward(src.nrows());
        self.encoder.forward(&(src + pos_encoding))
    }
    
    pub fn decode(&self, tgt: &DMatrix<f64>, encoder_output: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let pos_encoding = self.positional_encoding.forward(tgt.nrows());
        self.decoder.forward(&(tgt + pos_encoding), encoder_output)
    }
    
    pub fn forward(&self, src: &DMatrix<f64>, tgt: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let encoder_output = self.encode(src)?;
        let decoder_output = self.decode(tgt, &encoder_output)?;
        Ok(self.output_layer.forward(&decoder_output))
    }
}
