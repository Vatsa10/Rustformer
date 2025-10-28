use candle_core::{Tensor, Device, Result, IndexOp};
use crate::{Transformer, data::TranslationDataset};

/// Generate translation using greedy decoding
pub fn generate_greedy(
    model: &Transformer,
    src: &Tensor,
    dataset: &TranslationDataset,
    max_length: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut generated = vec![dataset.sos_token];
    
    for _ in 0..max_length {
        // Create target tensor from generated tokens
        let tgt_tensor = Tensor::from_vec(
            generated.clone(),
            (1, generated.len()),
            device,
        )?;
        
        // Forward pass
        let logits = model.forward(src, &tgt_tensor)?;
        
        // Get the last token's logits
        let last_logits = logits.i((0, generated.len() - 1))?;
        
        // Get the token with highest probability (greedy)
        let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;
        
        // Stop if we generate EOS token
        if next_token == dataset.eos_token {
            break;
        }
        
        generated.push(next_token);
    }
    
    Ok(generated)
}

/// Generate translation using beam search
pub fn generate_beam_search(
    model: &Transformer,
    src: &Tensor,
    dataset: &TranslationDataset,
    beam_width: usize,
    max_length: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    #[derive(Clone, Debug)]
    struct Beam {
        tokens: Vec<u32>,
        score: f32,
    }
    
    let mut beams = vec![Beam {
        tokens: vec![dataset.sos_token],
        score: 0.0,
    }];
    
    for _ in 0..max_length {
        let mut all_candidates = Vec::new();
        
        for beam in &beams {
            // Check if beam is complete
            if beam.tokens.last() == Some(&dataset.eos_token) {
                all_candidates.push(beam.clone());
                continue;
            }
            
            // Create target tensor
            let tgt_tensor = Tensor::from_vec(
                beam.tokens.clone(),
                (1, beam.tokens.len()),
                device,
            )?;
            
            // Forward pass
            let logits = model.forward(src, &tgt_tensor)?;
            
            // Get the last token's logits
            let last_logits = logits.i((0, beam.tokens.len() - 1))?;
            
            // Apply log softmax
            let log_probs = candle_nn::ops::log_softmax(&last_logits, 0)?;
            
            // Get top-k tokens
            let log_probs_vec = log_probs.to_vec1::<f32>()?;
            let mut indexed_probs: Vec<(usize, f32)> = log_probs_vec
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Expand beam with top-k candidates
            for (token_idx, log_prob) in indexed_probs.iter().take(beam_width) {
                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(*token_idx as u32);
                
                all_candidates.push(Beam {
                    tokens: new_tokens,
                    score: beam.score + log_prob,
                });
            }
        }
        
        // Keep top beam_width beams
        all_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beams = all_candidates.into_iter().take(beam_width).collect();
        
        // Check if all beams are complete
        if beams.iter().all(|b| b.tokens.last() == Some(&dataset.eos_token)) {
            break;
        }
    }
    
    // Return the best beam
    Ok(beams[0].tokens.clone())
}

/// Decode tokens to text
pub fn decode_tokens(tokens: &[u32], dataset: &TranslationDataset) -> String {
    tokens
        .iter()
        .filter(|&&t| t != dataset.sos_token && t != dataset.eos_token && t != dataset.pad_token)
        .filter_map(|&t| dataset.reverse_vocab.get(&t))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Encode text to tokens
pub fn encode_text(text: &str, dataset: &TranslationDataset) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|word| dataset.vocab.get(word).copied())
        .collect()
}

/// Generate and display translation sample
pub fn generate_sample(
    model: &Transformer,
    dataset: &TranslationDataset,
    src_text: &str,
    use_beam_search: bool,
    beam_width: usize,
    max_length: usize,
    device: &Device,
) -> Result<String> {
    // Encode source text
    let src_tokens = encode_text(src_text, dataset);
    
    if src_tokens.is_empty() {
        return Ok("[Unknown words]".to_string());
    }
    
    // Create source tensor
    let src_tensor = Tensor::from_vec(
        src_tokens.clone(),
        (1, src_tokens.len()),
        device,
    )?;
    
    // Generate translation
    let generated_tokens = if use_beam_search {
        generate_beam_search(model, &src_tensor, dataset, beam_width, max_length, device)?
    } else {
        generate_greedy(model, &src_tensor, dataset, max_length, device)?
    };
    
    // Decode to text
    Ok(decode_tokens(&generated_tokens, dataset))
}
