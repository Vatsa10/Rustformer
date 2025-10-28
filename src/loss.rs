use candle_core::{Tensor, Result, DType, IndexOp, D};

/// Calculates the cross-entropy loss with label smoothing.
///
/// # Arguments
///
/// * `logits` - The model's raw output logits. Shape: (batch_size * seq_len, num_classes).
/// * `labels` - The ground truth labels. Shape: (batch_size * seq_len).
/// * `num_classes` - The total number of classes in the vocabulary.
/// * `smoothing` - The label smoothing factor (e.g., 0.1).
///
/// # Returns
///
/// A scalar tensor representing the smoothed loss.
pub fn cross_entropy_with_smoothing(
    logits: &Tensor,
    labels: &Tensor,
    num_classes: usize,
    smoothing: f32,
) -> Result<Tensor> {
    let (batch_size, _) = logits.dims2()?;
    
    // Compute log probabilities
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
    
    if smoothing > 0.0 {
        // Label smoothing: distribute some probability mass to all classes
        let confidence = 1.0 - smoothing;
        let smoothing_value = smoothing / (num_classes - 1) as f32;
        
        // Create one-hot encoding for true labels
        let labels_u32 = labels.to_dtype(DType::U32)?;
        let mut nll_values = Vec::new();
        
        // Gather log probabilities for true labels
        for i in 0..batch_size {
            let label_idx = labels_u32.i(i)?.to_scalar::<u32>()? as usize;
            let log_prob = log_probs.i((i, label_idx))?.to_scalar::<f32>()?;
            nll_values.push(log_prob);
        }
        
        let nll_loss = Tensor::from_vec(nll_values, (batch_size,), logits.device())?;
        
        // Compute smoothed loss
        let smooth_loss = log_probs.sum(D::Minus1)?;
        let confidence_tensor = Tensor::new(&[-confidence], logits.device())?;
        let smoothing_tensor = Tensor::new(&[-smoothing_value], logits.device())?;
        
        let loss = (nll_loss.broadcast_mul(&confidence_tensor)? + smooth_loss.broadcast_mul(&smoothing_tensor)?)?;
        loss.mean(D::Minus1)
    } else {
        // Standard cross-entropy without smoothing
        candle_nn::loss::cross_entropy(logits, labels)
    }
}