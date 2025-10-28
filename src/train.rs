use anyhow::Result;
use std::collections::HashMap;
use candle_core::{Device, Var};
use candle_nn::{VarBuilder, Optimizer, AdamW, VarMap, optim::ParamsAdamW};
use chrono::Utc;

use crate::{
    ModelArgs, Transformer, MetricsTracker, ExperimentMonitor, loss, data, checkpoint, inference
};

/// Compute the L2 norm of gradients
fn compute_gradient_norm(_vars: &[Var]) -> Result<f64> {
    // Note: Candle's gradient computation is handled internally by the optimizer
    // This is a placeholder that returns a dummy value
    Ok(0.0)
}

/// Clip gradients by global norm
fn clip_gradients(_vars: &[Var], _max_norm: f64) -> Result<()> {
    // Note: Gradient clipping in Candle is handled by the optimizer
    // This is a placeholder function
    Ok(())
}

/// Main training function
pub fn train(args: &ModelArgs) -> Result<()> {
    println!("Setting up training environment...");

    let device = Device::Cpu;

    // Initialize dataset and update model args
    let dataset = data::TranslationDataset::new_dummy(&device)?;
    let mut args = args.clone();
    args.vocab_size = dataset.vocab_size;
    
    // Split dataset into train and validation (80/20 split)
    let total_pairs = dataset.pairs.len();
    let train_size = (total_pairs as f32 * 0.8) as usize;
    let train_indices: Vec<usize> = (0..train_size).collect();
    let val_indices: Vec<usize> = (train_size..total_pairs).collect();

    let varmap = VarMap::new();

    // Initialize model
    let model = Transformer::new(args.clone(), &varmap)?;
    println!("Transformer model initialized with vocab size: {}", args.vocab_size);

    // Optimizer
    let params = model.all_vars();
    let mut optimizer = AdamW::new(params.clone(), ParamsAdamW {
        lr: args.max_lr,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.epsilon,
        weight_decay: args.weight_decay,
    })?;
    println!("AdamW optimizer configured.");

    // Metrics and monitoring
    let mut metrics = MetricsTracker::new(10000);
    let mut monitor = ExperimentMonitor::new("transformer-training", "demo-run");
    monitor.log_config(&args);
    println!("Metrics and monitoring are set up.");

    println!("\nStarting training...\n");
    println!("Train samples: {}, Validation samples: {}", train_indices.len(), val_indices.len());
    let checkpoint_dir = "checkpoints";

    for step in 0..100 { // Simulate for 100 steps
        // Sample batch from training set
        let batch_indices: Vec<usize> = train_indices
            .iter()
            .cycle()
            .skip(step * args.batch_size)
            .take(args.batch_size.min(train_indices.len()))
            .copied()
            .collect();
        // Get a batch of data
        let (src, tgt, labels) = dataset.get_batch(&batch_indices, args.block_size, &device)?;

        // Forward pass
        let logits = model.forward(&src, &tgt)?;
        let loss = loss::cross_entropy_with_smoothing(&logits.flatten_to(1)?, &labels.flatten_to(1)?, args.vocab_size, args.label_smoothing as f32)?;

        let loss_val = loss.to_scalar::<f32>()? as f64;

        // Backward pass and optimization
        optimizer.backward_step(&loss)?;
        
        // Compute gradient norm (placeholder)
        let grad_norm = compute_gradient_norm(&params)?;

        // Update learning rate
        let current_lr = args.get_learning_rate(step);
        optimizer.set_learning_rate(current_lr);

        // Log metrics
        metrics.log_train_loss(loss_val);
        metrics.log_lr(current_lr);
        metrics.log_perplexity(loss_val);
        metrics.log_gradient_norm(grad_norm);

        let mut step_metrics = HashMap::new();
        step_metrics.insert("train_loss".to_string(), loss_val);
        step_metrics.insert("learning_rate".to_string(), current_lr);
        step_metrics.insert("perplexity".to_string(), loss_val.exp());
        step_metrics.insert("gradient_norm".to_string(), grad_norm);
        monitor.log_metrics(step_metrics, step);

        if step % 10 == 0 {
            println!("Step {:3} | Loss: {:.4} | LR: {:.2e} | Perplexity: {:.2} | Grad Norm: {:.4}",
                step, loss_val, current_lr, loss_val.exp(), grad_norm);
        }
        
        // Validation evaluation periodically
        if step > 0 && step % args.eval_interval == 0 && !val_indices.is_empty() {
            println!("\n📊 Running validation...");
            let val_batch_size = args.batch_size.min(val_indices.len());
            let val_batch: Vec<usize> = val_indices.iter().take(val_batch_size).copied().collect();
            
            let (val_src, val_tgt, val_labels) = dataset.get_batch(&val_batch, args.block_size, &device)?;
            let val_logits = model.forward(&val_src, &val_tgt)?;
            let val_loss = loss::cross_entropy_with_smoothing(
                &val_logits.flatten_to(1)?,
                &val_labels.flatten_to(1)?,
                args.vocab_size,
                args.label_smoothing as f32,
            )?;
            let val_loss_val = val_loss.to_scalar::<f32>()? as f64;
            
            metrics.log_val_loss(val_loss_val);
            println!("   Validation Loss: {:.4} | Perplexity: {:.2}", val_loss_val, val_loss_val.exp());
            
            let mut val_metrics = HashMap::new();
            val_metrics.insert("val_loss".to_string(), val_loss_val);
            val_metrics.insert("val_perplexity".to_string(), val_loss_val.exp());
            monitor.log_metrics(val_metrics, step);
        }
        
        // Generate sample translations periodically
        if step > 0 && step % args.generate_samples_interval == 0 {
            println!("\n🎯 Sample Translations (Step {}):", step);
            
            // Generate a few samples from the dataset
            for i in 0..2.min(dataset.pairs.len()) {
                let (src_tokens, tgt_tokens) = &dataset.pairs[i];
                
                // Decode source
                let src_text = inference::decode_tokens(src_tokens, &dataset);
                
                // Decode reference
                let ref_text = inference::decode_tokens(tgt_tokens, &dataset);
                
                // Generate translation
                if let Ok(generated) = inference::generate_sample(
                    &model,
                    &dataset,
                    &src_text,
                    false, // Use greedy for speed
                    1,
                    args.block_size,
                    &device,
                ) {
                    println!("  Source:    {}", src_text);
                    println!("  Reference: {}", ref_text);
                    println!("  Generated: {}", generated);
                    println!();
                }
            }
        }
        
        // Save checkpoint periodically
        if step > 0 && step % args.save_interval == 0 {
            let metadata = checkpoint::CheckpointMetadata {
                step,
                loss: loss_val,
                learning_rate: current_lr,
                model_args: args.clone(),
                timestamp: Utc::now().to_rfc3339(),
            };
            checkpoint::save_checkpoint(&varmap, metadata, checkpoint_dir)?;
            checkpoint::cleanup_old_checkpoints(checkpoint_dir, 3)?; // Keep last 3 checkpoints
        }
    }

    println!("\nTraining finished.\n");
    metrics.print_summary(100);
    monitor.save_to_json("experiment_results.json")?;

    Ok(())
}
