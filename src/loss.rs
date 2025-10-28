use candle_core::{Tensor, Result, DType};

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
    let confidence = 1.0 - smoothing;
    let smooth_dist = Tensor::full(smoothing / (num_classes - 1) as f32, (num_classes,), logits.device())?;

    // Create the smoothed target distribution
    let mut one_hot = Tensor::zeros(labels.shape(), DType::F32, logits.device())?;
    one_hot = one_hot.scatter_add(&labels.unsqueeze(1)?, &Tensor::ones_like(&labels.to_dtype(DType::F32)?)?, 1)?;
    
    let smoothed_targets = one_hot.broadcast_mul(&Tensor::full(confidence, (num_classes,), logits.device())?)?.broadcast_add(&smooth_dist)?;

    // Calculate the loss
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
    let loss = (smoothed_targets * log_probs)?.sum_all()? / (labels.shape().elem_count() as f64);

    Ok(loss.neg()?)
}
