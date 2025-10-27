//! # Transformer Architecture in Rust
//! 
//! Complete implementation of the Transformer architecture from "Attention Is All You Need" paper
//! by Vaswani et al. (2017) with advanced training features.
//! 
//! ## Features
//! 
//! ### Core Architecture
//! - Encoder with full multi-head self-attention
//! - Decoder with masked self-attention and cross-attention
//! - Sinusoidal positional embeddings
//! - Layer normalization and feed-forward networks
//! 
//! ### Training & Optimization
//! - AdamW optimizer with weight decay
//! - Learning rate warmup and cosine decay
//! - Gradient clipping by global norm
//! - Mixed precision training support
//! 
//! ### Generation Methods
//! - Top-K sampling with temperature
//! - Beam search
//! - Nucleus (top-p) sampling
//! - Repetition penalty
//! 
//! ### Monitoring
//! - Comprehensive metrics tracking
//! - Loss, perplexity, gradient norms
//! - Experiment tracking (WandB-like)
//! - Generation sample logging

pub mod config;
pub mod transformer;
pub mod model_args;
pub mod optimizer;
pub mod metrics;
pub mod generation;
pub mod monitoring;
pub mod attention;
pub mod complete_transformer;

pub use config::TransformerConfig;
pub use transformer::Transformer;
pub use model_args::ModelArgs;
pub use optimizer::{AdamW, clip_gradients};
pub use metrics::MetricsTracker;
pub use generation::{TopKSampler, BeamSearch, NucleusSampler};
pub use monitoring::ExperimentMonitor;
pub use complete_transformer::CompleteTransformer;