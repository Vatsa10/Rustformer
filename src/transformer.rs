use candle_core::{Tensor, Device, Result};
use candle_nn::{VarBuilder, Module, LayerNorm, Linear, layer_norm, linear};
use crate::attention::{MultiHeadAttention, MaskedMultiHeadAttention, CrossMultiHeadAttention};
use crate::model_args::ModelArgs;

// --- Position-wise Feed-Forward Network ---
pub struct PositionwiseFeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl PositionwiseFeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(d_model, d_ff, vb.pp("ffn_linear1"))?;
        let linear2 = linear(d_ff, d_model, vb.pp("ffn_linear2"))?;
        Ok(Self { linear1, linear2 })
    }
}
impl Module for PositionwiseFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear2.forward(&self.linear1.forward(x)?.relu()?)
    }
}

// --- Positional Encoding ---
pub struct PositionalEncoding {
    encoding: Tensor,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize, device: &Device) -> Result<Self> {
        let mut pe = Tensor::zeros((max_seq_len, d_model), candle_core::DType::F32, device)?;
        for pos in 0..max_seq_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (-(i as f32 / d_model as f32) * (10000.0_f32).ln()).exp();
                let angle = pos as f32 * div_term;
                pe = pe.slice_assign(&[pos..pos+1, i..i+1], &Tensor::from_slice(&[angle.sin()], (1,1), device)?)?;
                if i + 1 < d_model {
                    pe = pe.slice_assign(&[pos..pos+1, i+1..i+2], &Tensor::from_slice(&[angle.cos()], (1,1), device)?)?;
                }
            }
        }
        Ok(Self { encoding: pe })
    }

    pub fn forward(&self, seq_len: usize) -> Result<Tensor> {
        self.encoding.narrow(0, 0, seq_len)
    }
}

// --- Encoder Layer ---
pub struct EncoderLayer {
    self_attention: MultiHeadAttention,
    ffn: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(args.embeddings_dims, args.no_of_heads, vb.pp("self_attn"))?;
        let ffn = PositionwiseFeedForward::new(args.embeddings_dims, args.d_ff, vb.pp("ffn"))?;
        let norm1 = layer_norm(args.embeddings_dims, 1e-6, vb.pp("ln1"))?;
        let norm2 = layer_norm(args.embeddings_dims, 1e-6, vb.pp("ln2"))?;
        Ok(Self { self_attention, ffn, norm1, norm2 })
    }
}
impl Module for EncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x_attn = self.self_attention.forward(x)?;
        let x = self.norm1.forward(&(x_attn + residual)?)?;

        let residual = &x;
        let x_ffn = self.ffn.forward(&x)?;
        let x = self.norm2.forward(&(x_ffn + residual)?)?;
        Ok(x)
    }
}

// --- Encoder ---
pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..args.no_of_encoder_layers {
            layers.push(EncoderLayer::new(args, vb.pp(&format!("layer_{}", i)))?);
        }
        Ok(Self { layers })
    }
}

impl Module for Encoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
    pub fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let masked_self_attention = MaskedMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads, vb.pp("masked_self_attn"))?;
        let cross_attention = CrossMultiHeadAttention::new(args.embeddings_dims, args.no_of_heads, vb.pp("cross_attn"))?;
        let ffn = PositionwiseFeedForward::new(args.embeddings_dims, args.d_ff, vb.pp("ffn"))?;
        let norm1 = layer_norm(args.embeddings_dims, 1e-6, vb.pp("ln1"))?;
        let norm2 = layer_norm(args.embeddings_dims, 1e-6, vb.pp("ln2"))?;
        let norm3 = layer_norm(args.embeddings_dims, 1e-6, vb.pp("ln3"))?;
        Ok(Self { masked_self_attention, cross_attention, ffn, norm1, norm2, norm3 })
    }

    pub fn forward(&self, x: &Tensor, encoder_output: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x_attn = self.masked_self_attention.forward(x)?;
        let x = self.norm1.forward(&(x_attn + residual)?)?;

        let residual = &x;
        let x_cross = self.cross_attention.forward(&x, encoder_output)?;
        let x = self.norm2.forward(&(x_cross + residual)?)?;

        let residual = &x;
        let x_ffn = self.ffn.forward(&x)?;
        let x = self.norm3.forward(&(x_ffn + residual)?)?;
        Ok(x)
    }
}

// --- Decoder ---
pub struct Decoder {
    layers: Vec<DecoderLayer>,
}

impl Decoder {
    pub fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..args.no_of_decoder_layers {
            layers.push(DecoderLayer::new(args, vb.pp(&format!("layer_{}", i)))?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor, encoder_output: &Tensor) -> Result<Tensor> {
        let mut x_out = x.clone();
        for layer in &self.layers {
            x_out = layer.forward(&x_out, encoder_output)?;
        }
        Ok(x_out)
    }
}

// --- Complete Transformer ---
pub struct Transformer {
    embedding: candle_nn::Embedding,
    encoder: Encoder,
    decoder: Decoder,
    positional_encoding: PositionalEncoding,
    output_layer: Linear,
    args: ModelArgs,
}

impl Transformer {
    pub fn new(args: ModelArgs, vb: VarBuilder) -> Result<Self> {
        args.validate().map_err(|e| candle_core::Error::Msg(e))?;
        let device = vb.device();
        Ok(Self {
            embedding: candle_nn::embedding(args.vocab_size, args.embeddings_dims, vb.pp("embedding"))?,
            encoder: Encoder::new(&args, vb.pp("encoder"))?,
            decoder: Decoder::new(&args, vb.pp("decoder"))?,
            positional_encoding: PositionalEncoding::new(args.block_size, args.embeddings_dims, device)?,
            output_layer: linear(args.embeddings_dims, args.vocab_size, vb.pp("output"))?,
            args,
        })
    }
    
    pub fn forward(&self, src: &Tensor, tgt: &Tensor) -> Result<Tensor> {
        let src_embed = self.embedding.forward(src)?;
        let tgt_embed = self.embedding.forward(tgt)?;

        let src_pe = self.positional_encoding.forward(src.dim(1)?)?;
        let tgt_pe = self.positional_encoding.forward(tgt.dim(1)?)?;

        let encoder_output = self.encoder.forward(&(src_embed + src_pe)?)?;
        let decoder_output = self.decoder.forward(&(tgt_embed + tgt_pe)?, &encoder_output)?;
        self.output_layer.forward(&decoder_output)
    }
}