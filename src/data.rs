use candle_core::{Tensor, Device, Result};
use std::collections::HashMap;

const SOS_TOKEN: &str = "<sos>";
const EOS_TOKEN: &str = "<eos>";
const PAD_TOKEN: &str = "<pad>";

/// Represents a tokenized dataset with vocabulary
pub struct TranslationDataset {
    pub vocab: HashMap<String, u32>,
    pub reverse_vocab: HashMap<u32, String>,
    pub pairs: Vec<(Vec<u32>, Vec<u32>)>, // (source, target)
    pub vocab_size: usize,
    pub sos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
}

impl TranslationDataset {
    /// Creates a tiny simulated dataset for demonstration purposes.
    pub fn new_dummy(_device: &Device) -> Result<Self> {
        let pairs = vec![
            ("hello world", "hallo welt"),
            ("good morning", "guten morgen"),
            ("how are you", "wie geht es ihnen"),
            ("thank you", "danke schön"),
        ];

        let mut vocab = HashMap::new();
        vocab.insert(PAD_TOKEN.to_string(), 0);
        vocab.insert(SOS_TOKEN.to_string(), 1);
        vocab.insert(EOS_TOKEN.to_string(), 2);

        for (src, tgt) in &pairs {
            for word in src.split_whitespace() {
                if !vocab.contains_key(word) {
                    vocab.insert(word.to_string(), vocab.len() as u32);
                }
            }
            for word in tgt.split_whitespace() {
                if !vocab.contains_key(word) {
                    vocab.insert(word.to_string(), vocab.len() as u32);
                }
            }
        }

        let reverse_vocab = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        let vocab_size = vocab.len();
        let sos_token = vocab[SOS_TOKEN];
        let eos_token = vocab[EOS_TOKEN];
        let pad_token = vocab[PAD_TOKEN];

        let tokenized_pairs = pairs
            .iter()
            .map(|(src, tgt)| {
                let src_tokens = src.split_whitespace().map(|w| vocab[w]).collect();
                let tgt_tokens = tgt.split_whitespace().map(|w| vocab[w]).collect();
                (src_tokens, tgt_tokens)
            })
            .collect();

        Ok(Self {
            vocab,
            reverse_vocab,
            pairs: tokenized_pairs,
            vocab_size,
            sos_token,
            eos_token,
            pad_token,
        })
    }

    /// Creates a batch of padded tensors for training.
    pub fn get_batch(&self, indices: &[usize], block_size: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let mut src_batch = Vec::new();
        let mut tgt_batch = Vec::new();
        let mut label_batch = Vec::new();

        for &i in indices {
            let (src, tgt) = &self.pairs[i];

            // Prepare source tensor
            let mut src_padded = vec![self.pad_token; block_size];
            let src_len = src.len().min(block_size);
            src_padded[..src_len].copy_from_slice(&src[..src_len]);
            src_batch.extend_from_slice(&src_padded);

            // Prepare target tensor (decoder input)
            let mut tgt_padded = vec![self.pad_token; block_size];
            tgt_padded[0] = self.sos_token;
            let tgt_len = tgt.len().min(block_size - 1);
            tgt_padded[1..=tgt_len].copy_from_slice(&tgt[..tgt_len]);
            tgt_batch.extend_from_slice(&tgt_padded);

            // Prepare label tensor (decoder output)
            let mut label_padded = vec![self.pad_token; block_size];
            let label_len = tgt.len().min(block_size - 1);
            label_padded[..label_len].copy_from_slice(&tgt[..label_len]);
            label_padded[label_len] = self.eos_token;
            label_batch.extend_from_slice(&label_padded);
        }

        let src_tensor = Tensor::from_vec(src_batch, (indices.len(), block_size), device)?;
        let tgt_tensor = Tensor::from_vec(tgt_batch, (indices.len(), block_size), device)?;
        let label_tensor = Tensor::from_vec(label_batch, (indices.len(), block_size), device)?;

        Ok((src_tensor, tgt_tensor, label_tensor))
    }
}