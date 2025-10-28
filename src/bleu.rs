use std::collections::HashMap;

/// Calculate BLEU score for machine translation evaluation
/// Based on the paper "BLEU: a Method for Automatic Evaluation of Machine Translation" (Papineni et al., 2002)
pub struct BleuScore {
    max_n: usize,
}

impl BleuScore {
    pub fn new(max_n: usize) -> Self {
        Self { max_n }
    }
    
    /// Calculate BLEU score between reference and candidate translations
    pub fn calculate(&self, reference: &[String], candidate: &[String]) -> f64 {
        if candidate.is_empty() {
            return 0.0;
        }
        
        // Calculate brevity penalty
        let ref_len = reference.len() as f64;
        let cand_len = candidate.len() as f64;
        let brevity_penalty = if cand_len < ref_len {
            (1.0 - ref_len / cand_len).exp()
        } else {
            1.0
        };
        
        // Calculate precision for each n-gram order
        let mut precisions = Vec::new();
        for n in 1..=self.max_n {
            let precision = self.calculate_ngram_precision(reference, candidate, n);
            if precision > 0.0 {
                precisions.push(precision.ln());
            } else {
                // If any n-gram precision is 0, BLEU is 0
                return 0.0;
            }
        }
        
        // Geometric mean of precisions
        let avg_log_precision = precisions.iter().sum::<f64>() / precisions.len() as f64;
        
        brevity_penalty * avg_log_precision.exp()
    }
    
    /// Calculate n-gram precision
    fn calculate_ngram_precision(&self, reference: &[String], candidate: &[String], n: usize) -> f64 {
        if candidate.len() < n {
            return 0.0;
        }
        
        // Get n-grams from reference
        let ref_ngrams = self.get_ngrams(reference, n);
        
        // Get n-grams from candidate
        let cand_ngrams = self.get_ngrams(candidate, n);
        
        // Count matches (clipped)
        let mut matches = 0;
        for (ngram, cand_count) in &cand_ngrams {
            if let Some(&ref_count) = ref_ngrams.get(ngram) {
                matches += cand_count.min(&ref_count);
            }
        }
        
        let total_cand_ngrams: usize = cand_ngrams.values().sum();
        
        if total_cand_ngrams == 0 {
            0.0
        } else {
            matches as f64 / total_cand_ngrams as f64
        }
    }
    
    /// Extract n-grams from a sequence
    fn get_ngrams(&self, tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
        let mut ngrams = HashMap::new();
        
        if tokens.len() < n {
            return ngrams;
        }
        
        for i in 0..=tokens.len() - n {
            let ngram = tokens[i..i + n].to_vec();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }
        
        ngrams
    }
    
    /// Calculate corpus-level BLEU score
    pub fn calculate_corpus_bleu(&self, references: &[Vec<String>], candidates: &[Vec<String>]) -> f64 {
        if references.len() != candidates.len() || references.is_empty() {
            return 0.0;
        }
        
        let mut total_ref_len = 0.0;
        let mut total_cand_len = 0.0;
        let mut total_matches = vec![0; self.max_n];
        let mut total_possible = vec![0; self.max_n];
        
        for (reference, candidate) in references.iter().zip(candidates.iter()) {
            total_ref_len += reference.len() as f64;
            total_cand_len += candidate.len() as f64;
            
            for n in 1..=self.max_n {
                let ref_ngrams = self.get_ngrams(reference, n);
                let cand_ngrams = self.get_ngrams(candidate, n);
                
                let mut matches = 0;
                for (ngram, cand_count) in &cand_ngrams {
                    if let Some(&ref_count) = ref_ngrams.get(ngram) {
                        matches += cand_count.min(&ref_count);
                    }
                }
                
                total_matches[n - 1] += matches;
                total_possible[n - 1] += cand_ngrams.values().sum::<usize>();
            }
        }
        
        // Calculate brevity penalty
        let brevity_penalty = if total_cand_len < total_ref_len {
            (1.0 - total_ref_len / total_cand_len).exp()
        } else {
            1.0
        };
        
        // Calculate geometric mean of precisions
        let mut log_precisions = Vec::new();
        for n in 0..self.max_n {
            if total_possible[n] > 0 && total_matches[n] > 0 {
                let precision = total_matches[n] as f64 / total_possible[n] as f64;
                log_precisions.push(precision.ln());
            } else {
                return 0.0;
            }
        }
        
        let avg_log_precision = log_precisions.iter().sum::<f64>() / log_precisions.len() as f64;
        
        brevity_penalty * avg_log_precision.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perfect_match() {
        let bleu = BleuScore::new(4);
        let reference = vec!["the", "cat", "sat", "on", "the", "mat"]
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let candidate = reference.clone();
        
        let score = bleu.calculate(&reference, &candidate);
        assert!((score - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_no_match() {
        let bleu = BleuScore::new(4);
        let reference = vec!["the", "cat", "sat"]
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let candidate = vec!["dog", "runs", "fast"]
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        
        let score = bleu.calculate(&reference, &candidate);
        assert_eq!(score, 0.0);
    }
    
    #[test]
    fn test_partial_match() {
        let bleu = BleuScore::new(4);
        let reference = vec!["the", "cat", "sat", "on", "the", "mat"]
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let candidate = vec!["the", "cat", "is", "on", "the", "mat"]
            .iter().map(|s| s.to_string()).collect::<Vec<_>>();
        
        let score = bleu.calculate(&reference, &candidate);
        assert!(score > 0.0 && score < 1.0);
    }
}
