use anyhow::Result;
use std::collections::HashMap;
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, Optimizer, loss, VarMap, optim::{AdamW, ParamsAdamW}};

use crate::{
    ModelArgs, Transformer, MetricsTracker, ExperimentMonitor
};

/// Main training function
pub fn train(args: &ModelArgs) -> Result<()> {
    println!("Setting up training environment...");

    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

    // Initialize model
    let model = Transformer::new(args.clone(), vb)?;
    println!("Transformer model initialized.");

    // Optimizer
    let config = ParamsAdamW {
        lr: args.max_lr,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.epsilon,
        weight_decay: args.weight_decay,
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), config)?;
    println!("AdamW optimizer configured.");

    // Metrics and monitoring
    let mut metrics = MetricsTracker::new(10000);
    let mut monitor = ExperimentMonitor::new("transformer-training", "demo-run");
    monitor.log_config(args);
    println!("Metrics and monitoring are set up.");

    println!("\nStarting training...\n");
    for step in 0..100 { // Simulate for 100 steps
        // Dummy data (U32 token IDs)
        let src = Tensor::ones((args.batch_size, args.block_size), candle_core::DType::U32, &device)?;
        let tgt = Tensor::ones((args.batch_size, args.block_size), candle_core::DType::U32, &device)?;
        let labels = Tensor::ones((args.batch_size, args.block_size), candle_core::DType::U32, &device)?;

        // Forward pass
        let logits = model.forward(&src, &tgt)?;
        let loss = loss::cross_entropy(&logits.flatten_to(1)?, &labels.flatten_to(1)?)?;

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