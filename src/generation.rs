use nalgebra::DMatrix;
use rand::Rng;

/// Top-K sampling with temperature and repetition penalty
pub struct TopKSampler {
    pub k: usize,
    pub temperature: f64,
    pub repetition_penalty: f64,
}

impl TopKSampler {
    pub fn new(k: usize, temperature: f64, repetition_penalty: f64) -> Self {
        Self {
            k,
            temperature,
            repetition_penalty,
        }
    }
    
    /// Sample next token using top-k sampling
    pub fn sample(&self, logits: &DMatrix<f64>, generated_tokens: &[usize]) -> usize {
        let mut rng = rand::thread_rng();
        
        // Apply repetition penalty
        let mut penalized_logits = logits.clone();
        for &token in generated_tokens {
            if token < penalized_logits.ncols() {
                penalized_logits[(0, token)] /= self.repetition_penalty;
            }
        }
        
        // Apply temperature
        let scaled_logits = penalized_logits.map(|x| x / self.temperature);
        
        // Convert to probabilities (softmax)
        let max_logit = scaled_logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits = scaled_logits.map(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.iter().sum::<f64>();
        let probs = exp_logits.map(|x| x / sum_exp);
        
        // Get top-k indices
        let mut indexed_probs: Vec<(usize, f64)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_k = indexed_probs.into_iter().take(self.k).collect::<Vec<_>>();
        
        // Renormalize top-k probabilities
        let top_k_sum: f64 = top_k.iter().map(|(_, p)| p).sum();
        let normalized_probs: Vec<(usize, f64)> = top_k.iter()
            .map(|(idx, p)| (*idx, p / top_k_sum))
            .collect();
        
        // Sample from top-k distribution
        let rand_val: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (idx, prob) in normalized_probs {
            cumsum += prob;
            if rand_val <= cumsum {
                return idx;
            }
        }
        
        // Fallback to most likely token
        top_k[0].0
    }
}

/// Beam search for better generation quality
#[derive(Clone, Debug)]
pub struct Beam {
    pub tokens: Vec<usize>,
    pub score: f64,
}

pub struct BeamSearch {
    pub beam_width: usize,
    pub max_length: usize,
    pub repetition_penalty: f64,
}

impl BeamSearch {
    pub fn new(beam_width: usize, max_length: usize, repetition_penalty: f64) -> Self {
        Self {
            beam_width,
            max_length,
            repetition_penalty,
        }
    }
    
    /// Perform beam search to generate sequences
    /// This is a simplified version - in practice you'd call the model at each step
    pub fn search(&self, start_token: usize, vocab_size: usize) -> Vec<Beam> {
        let mut beams = vec![Beam {
            tokens: vec![start_token],
            score: 0.0,
        }];
        
        for _ in 0..self.max_length {
            let mut all_candidates = Vec::new();
            
            for beam in &beams {
                // In a real implementation, we would:
                // 1. Run the model to get logits for next token
                // 2. Apply repetition penalty
                // 3. Compute log probabilities
                // 4. Expand beam with top-k candidates
                
                // For demo purposes, we'll simulate this
                for token in 0..vocab_size.min(self.beam_width * 2) {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token);
                    
                    // Simplified scoring (would use actual model logits)
                    let mut score = beam.score - (token as f64 * 0.01);
                    
                    // Apply repetition penalty
                    if beam.tokens.contains(&token) {
                        score -= self.repetition_penalty.ln();
                    }
                    
                    all_candidates.push(Beam {
                        tokens: new_tokens,
                        score,
                    });
                }
            }
            
            // Keep top beam_width beams
            all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams = all_candidates.into_iter().take(self.beam_width).collect();
        }
        
        beams
    }
}

/// Nucleus (top-p) sampling as an alternative to top-k
pub struct NucleusSampler {
    pub p: f64,
    pub temperature: f64,
    pub repetition_penalty: f64,
}

impl NucleusSampler {
    pub fn new(p: f64, temperature: f64, repetition_penalty: f64) -> Self {
        Self {
            p,
            temperature,
            repetition_penalty,
        }
    }
    
    /// Sample using nucleus sampling (top-p)
    pub fn sample(&self, logits: &DMatrix<f64>, generated_tokens: &[usize]) -> usize {
        let mut rng = rand::thread_rng();
        
        // Apply repetition penalty and temperature (same as top-k)
        let mut penalized_logits = logits.clone();
        for &token in generated_tokens {
            if token < penalized_logits.ncols() {
                penalized_logits[(0, token)] /= self.repetition_penalty;
            }
        }
        
        let scaled_logits = penalized_logits.map(|x| x / self.temperature);
        
        // Softmax
        let max_logit = scaled_logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits = scaled_logits.map(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.iter().sum::<f64>();
        let probs = exp_logits.map(|x| x / sum_exp);
        
        // Sort by probability
        let mut indexed_probs: Vec<(usize, f64)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Find nucleus (cumulative probability <= p)
        let mut cumsum = 0.0;
        let mut nucleus = Vec::new();
        for (idx, prob) in indexed_probs {
            cumsum += prob;
            nucleus.push((idx, prob));
            if cumsum >= self.p {
                break;
            }
        }
        
        // Renormalize and sample
        let nucleus_sum: f64 = nucleus.iter().map(|(_, p)| p).sum();
        let rand_val: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (idx, prob) in nucleus {
            cumsum += prob / nucleus_sum;
            if rand_val <= cumsum {
                return idx;
            }
        }
        
        // Fallback
        0
    }
}