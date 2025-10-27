use anyhow::Result;
use std::collections::HashMap;

use crate::{
    ModelArgs,
    Transformer,
    AdamW,
    MetricsTracker,
    ExperimentMonitor,
    data::Dataset,
    data::DataLoader
};

/// Main training function
pub fn train(args: &ModelArgs) -> Result<()> {
    println!("Setting up training environment...");

    // Initialize model
    let model = Transformer::new(args.clone())?;
    println!("Transformer model initialized.");

    // Optimizer
    let mut optimizer = AdamW::new(
        args.max_lr,
        args.beta1,
        args.beta2,
        args.epsilon,
        args.weight_decay,
    );
    println!("AdamW optimizer configured.");

    // Metrics and monitoring
    let mut metrics = MetricsTracker::new(10000);
    let mut monitor = ExperimentMonitor::new("transformer-training", "demo-run");
    monitor.log_config(args);
    println!("Metrics and monitoring are set up.");

    // Dataset and DataLoader
    let dataset = Dataset::new_dummy("dummy_dataset", args.vocab_size, 1000, args.block_size);
    let dataloader = DataLoader::new(&dataset, args.batch_size);
    println!("Dummy dataset and dataloader created.");

    println!("\nStarting training simulation...\n");
    for (epoch, (src, tgt)) in dataloader.enumerate() {
        if epoch >= 10 { // Simulate for 10 steps
            break;
        }
        let step = epoch * 100;

        // Forward pass
        let _output = model.forward(&src, &tgt)?;

        // Simulate loss calculation
        let loss = 4.5 - (epoch as f64 * 0.2);
        let grad_norm = 2.0 - (epoch as f64 * 0.1);

        // Update learning rate
        let current_lr = args.get_learning_rate(step);
        optimizer.set_lr(current_lr);

        // Log metrics
        metrics.log_train_loss(loss);
        metrics.log_lr(current_lr);
        metrics.log_gradient_norm(grad_norm);
        metrics.log_perplexity(loss);

        let mut step_metrics = HashMap::new();
        step_metrics.insert("train_loss".to_string(), loss);
        step_metrics.insert("learning_rate".to_string(), current_lr);
        step_metrics.insert("gradient_norm".to_string(), grad_norm);
        step_metrics.insert("perplexity".to_string(), loss.exp());
        monitor.log_metrics(step_metrics, step);

        if epoch % 2 == 0 {
            println!("Step {:3} | Loss: {:.4} | LR: {:.2e} | Grad Norm: {:.4} | Perplexity: {:.2}",
                step, loss, current_lr, grad_norm, loss.exp());
        }
    }

    println!("\nTraining simulation finished.\n");
    metrics.print_summary(10);
    monitor.save_to_json("experiment_results.json")?;

    Ok(())
}
