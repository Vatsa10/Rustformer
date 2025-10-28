use anyhow::Result;
use std::collections::HashMap;
use candle_core::Device;
use candle_nn::{VarBuilder, Optimizer, AdamW};

use crate::{
    ModelArgs, Transformer, MetricsTracker, ExperimentMonitor, loss, data
};

/// Main training function
pub fn train(args: &ModelArgs) -> Result<()> {
    println!("Setting up training environment...");

    let device = Device::Cpu;

    // Initialize dataset and update model args
    let dataset = data::TranslationDataset::new_dummy(&device)?;
    let mut args = args.clone();
    args.vocab_size = dataset.vocab_size;

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

    // Initialize model
    let model = Transformer::new(args.clone(), vb)?;
    println!("Transformer model initialized with vocab size: {}", args.vocab_size);

    // Optimizer
    let params = model.all_vars();
    let mut optimizer = AdamW::new(params, args.max_lr)?;
    println!("AdamW optimizer configured.");

    // Metrics and monitoring
    let mut metrics = MetricsTracker::new(10000);
    let mut monitor = ExperimentMonitor::new("transformer-training", "demo-run");
    monitor.log_config(&args);
    println!("Metrics and monitoring are set up.");

    println!("\nStarting training...\n");
    let batch_indices: Vec<usize> = (0..args.batch_size.min(dataset.pairs.len())).collect();

    for step in 0..100 { // Simulate for 100 steps
        // Get a batch of data
        let (src, tgt, labels) = dataset.get_batch(&batch_indices, args.block_size, &device)?;

        // Forward pass
        let logits = model.forward(&src, &tgt)?;
        let loss = loss::cross_entropy_with_smoothing(&logits.flatten_to(1)?, &labels.flatten_to(1)?, args.vocab_size, args.label_smoothing as f32)?;

        // Backward pass and optimization
        optimizer.backward_step(&loss)?;

        let loss_val = loss.to_scalar::<f32>()? as f64;

        // Update learning rate
        let current_lr = args.get_learning_rate(step);
        optimizer.set_learning_rate(current_lr);

        // Log metrics
        metrics.log_train_loss(loss_val);
        metrics.log_lr(current_lr);
        metrics.log_perplexity(loss_val);

        let mut step_metrics = HashMap::new();
        step_metrics.insert("train_loss".to_string(), loss_val);
        step_metrics.insert("learning_rate".to_string(), current_lr);
        step_metrics.insert("perplexity".to_string(), loss_val.exp());
        monitor.log_metrics(step_metrics, step);

        if step % 10 == 0 {
            println!("Step {:3} | Loss: {:.4} | LR: {:.2e} | Perplexity: {:.2}",
                step, loss_val, current_lr, loss_val.exp());
        }
    }

    println!("\nTraining finished.\n");
    metrics.print_summary(100);
    monitor.save_to_json("experiment_results.json")?;

    Ok(())
}