use candle_core::{Tensor, Device, Result};
use candle_nn::{VarBuilder, Module, linear, Linear};

// --- Scaled Dot-Product Attention ---
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let d_k = q.dim(q.dims().len() - 1)? as f64;
    let scores = (q.matmul(&k.t()?)? / d_k.sqrt())?;
    let scores = if let Some(mask) = mask {
        scores.broadcast_add(mask)?
    } else {
        scores
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
    attn_weights.matmul(v)
}

// --- Multi-Head Attention for Encoder ---
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    d_head: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / num_heads;
        let q_proj = linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self { q_proj, k_proj, v_proj, out_proj, num_heads, d_head })
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        x.reshape((batch_size, seq_len, self.num_heads, self.d_head))?.transpose(1, 2)
    }

    fn combine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, _, seq_len, _) = x.dims4()?;
        x.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.d_head))
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        let attn_output = scaled_dot_product_attention(&q, &k, &v, None)?;
        let combined = self.combine_heads(&attn_output)?;

        self.out_proj.forward(&combined)
    }
}


// --- Masked Multi-Head Attention for Decoder ---
pub struct MaskedMultiHeadAttention {
    mha: MultiHeadAttention,
}

impl MaskedMultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let mha = MultiHeadAttention::new(d_model, num_heads, vb)?;
        Ok(Self { mha })
    }

    fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. })
            })
            .collect();
        Tensor::from_vec(mask, (seq_len, seq_len), device)
    }
}

impl Module for MaskedMultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len, _d_model) = x.dims3()?;
        let q = self.mha.q_proj.forward(x)?;
        let k = self.mha.k_proj.forward(x)?;
        let v = self.mha.v_proj.forward(x)?;

        let q = self.mha.split_heads(&q)?;
        let k = self.mha.split_heads(&k)?;
        let v = self.mha.split_heads(&v)?;

        let mask = Self::create_causal_mask(seq_len, x.device())?;
        let attn_output = scaled_dot_product_attention(&q, &k, &v, Some(&mask))?;
        let combined = self.mha.combine_heads(&attn_output)?;

        self.mha.out_proj.forward(&combined)
    }
}

// --- Cross-Multi-Head Attention for Encoder-Decoder ---
pub struct CrossMultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    d_head: usize,
}

impl CrossMultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / num_heads;
        let q_proj = linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self { q_proj, k_proj, v_proj, out_proj, num_heads, d_head })
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        x.reshape((batch_size, seq_len, self.num_heads, self.d_head))?.transpose(1, 2)
    }

    fn combine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, _, seq_len, _) = x.dims4()?;
        x.transpose(1, 2)?.reshape((batch_size, seq_len, self.num_heads * self.d_head))
    }

    pub fn forward(&self, x: &Tensor, encoder_output: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(encoder_output)?;
        let v = self.v_proj.forward(encoder_output)?;

        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        let attn_output = scaled_dot_product_attention(&q, &k, &v, None)?;
        let combined = self.combine_heads(&attn_output)?;

        self.out_proj.forward(&combined)
    }
}
