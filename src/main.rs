use anyhow::Result;
use transformer_rust::{
    ModelArgs,
    train::train
};

fn main() -> Result<()> {
    println!("Transformer Implementation in Rust");
    println!("=====================================");

    // Initialize model configuration
    let args = ModelArgs::default();
    args.validate().map_err(|e| anyhow::anyhow!(e))?;

    print_configuration(&args);

    // Start training
    train(&args)?;

    println!("\n=====================================");
    println!("Run finished successfully!");

    Ok(())
}

fn print_configuration(args: &ModelArgs) {
    println!("Model Configuration:");
    println!();
    println!("  Architecture:");
    println!("    • Block size (max seq length): {}", args.block_size);
    println!("    • Embedding dimensions: {}", args.embeddings_dims);
    println!("    • Number of attention heads: {}", args.no_of_heads);
    println!("    • Encoder layers: {}", args.no_of_encoder_layers);
    println!("    • Decoder layers: {}", args.no_of_decoder_layers);
    println!("    • Feed-forward dimensions: {}", args.d_ff);
    println!("    • Vocabulary size: {}", args.vocab_size);
    println!();
    println!("  Optimization:");
    println!("    • Max learning rate: {}", args.max_lr);
    println!("    • Min learning rate: {}", args.min_lr);
    println!("    • Warmup steps: {}", args.warmup_steps);
    println!("    • Weight decay: {}", args.weight_decay);
    println!("    • Beta1: {}, Beta2: {}", args.beta1, args.beta2);
    println!("    • Gradient clip max norm: {}", args.grad_clip_max_norm);
    println!();
    println!("  Training:");
    println!("    • Batch size: {}", args.batch_size);
    println!("    • Dropout: {}", args.dropout);
    println!("    • Label smoothing: {}", args.label_smoothing);
    println!("    • Mixed precision: {}", args.use_mixed_precision);
    println!();
}
