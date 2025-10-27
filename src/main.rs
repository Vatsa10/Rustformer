use anyhow::Result;
use clap::Parser;
use transformer_rust::{train::train, ModelArgs};

fn main() -> Result<()> {
    println!("Transformer Implementation in Rust");
    println!("=====================================");

    // Initialize model configuration from command-line arguments
    let args = ModelArgs::parse();
    args.validate().map_err(|e| anyhow::anyhow!(e))?;

    print_configuration(&args);

    // Start training
    train(&args)?;

    println!("
--- Training Finished ---");
    println!("Placeholder for evaluation step (e.g., calculating BLEU score).");

    println!("
=====================================");
    println!("Run finished successfully!");

    Ok(())
}

fn print_configuration(args: &ModelArgs) {
    println!("Model Configuration:");
    println!();
    println!("  Architecture:");
    println!("    • Block size (max seq length): {}", args.block_size);
    println!("    • Embedding dimensions (d_model): {}", args.embeddings_dims);
    println!("    • Attention heads: {}", args.no_of_heads);
    println!("    • Encoder layers: {}", args.no_of_encoder_layers);
    println!("    • Decoder layers: {}", args.no_of_decoder_layers);
    println!("    • Feed-forward dimensions (d_ff): {}", args.d_ff);
    println!("    • Vocabulary size: {}", args.vocab_size);
    println!();
    println!("  Optimization:");
    println!("    • Max learning rate: {}", args.max_lr);
    println!("    • Min learning rate: {}", args.min_lr);
    println!("    • Warmup steps: {}", args.warmup_steps);
    println!("    • Weight decay: {}", args.weight_decay);
    println!("    • Adam (β1, β2): ({}, {})", args.beta1, args.beta2);
    println!("    • Adam (ε): {}", args.epsilon);
    println!("    • Gradient clip max norm: {}", args.grad_clip_max_norm);
    println!("    * Note: Using linear warmup + cosine decay schedule.");
    println!();
    println!("  Training:");
    println!("    • Batch size: {}", args.batch_size);
    println!("    • Dropout: {}", args.dropout);
    println!("    • Label smoothing: {}", args.label_smoothing);
    println!("    • Mixed precision (BF16): {}", args.use_mixed_precision);
    println!();
}
