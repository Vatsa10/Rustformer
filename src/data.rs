use nalgebra::DMatrix;
use std::collections::HashMap;

/// Represents a tokenized dataset with vocabulary
pub struct Dataset {
    pub name: String,
    pub tokenized_texts: Vec<Vec<usize>>,
    pub vocab: HashMap<String, usize>,
    pub reverse_vocab: HashMap<usize, String>,
    pub vocab_size: usize,
}

impl Dataset {
    /// Create a dummy dataset for demonstration
    pub fn new_dummy(name: &str, vocab_size: usize, num_samples: usize, max_len: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        for i in 0..vocab_size {
            let token = format!("<token_{}>", i);
            vocab.insert(token.clone(), i);
            reverse_vocab.insert(i, token);
        }

        let mut tokenized_texts = Vec::new();
        for i in 0..num_samples {
            let len = (i % max_len) + 1;
            tokenized_texts.push((0..len).map(|j| (i + j) % vocab_size).collect());
        }

        Self {
            name: name.to_string(),
            tokenized_texts,
            vocab,
            reverse_vocab,
            vocab_size,
        }
    }
}

/// Tokenizer for converting text to token IDs
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        Self { vocab }
    }

    /// Tokenize a sentence (simplified)
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| self.vocab.get(word).copied().unwrap_or(0)) // 0 for <unk>
            .collect()
    }
}

/// Data loader for creating batches from a dataset
pub struct DataLoader<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    cursor: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a Dataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            cursor: 0,
        }
    }
}

/// Iterator for the data loader
impl<'a> Iterator for DataLoader<'a> {
    type Item = (DMatrix<f64>, DMatrix<f64>); // (source_batch, target_batch)

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.dataset.tokenized_texts.len() {
            return None;
        }

        let end = (self.cursor + self.batch_size).min(self.dataset.tokenized_texts.len());
        let batch_texts = &self.dataset.tokenized_texts[self.cursor..end];
        self.cursor = end;

        // Dummy conversion to DMatrix for demonstration
        // In a real scenario, this would involve padding and embedding lookups
        let src_matrix = DMatrix::from_element(batch_texts.len(), 512, 0.1);
        let tgt_matrix = DMatrix::from_element(batch_texts.len(), 512, 0.1);

        Some((src_matrix, tgt_matrix))
    }
}
