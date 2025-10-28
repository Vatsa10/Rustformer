//! # Transformer Architecture in Rust
//! 
//! Complete implementation of the Transformer architecture from "Attention Is All You Need" paper
//! by Vaswani et al. (2017) with advanced training features.

pub mod config;
pub mod transformer;
pub mod model_args;
pub mod optimizer;
pub mod metrics;
pub mod generation;
pub mod monitoring;
pub mod attention;
pub mod data;
pub mod train;
pub mod loss;
pub mod bleu;
pub mod checkpoint;
pub mod inference;

// This file is intentionally left blank as its content has been merged into transformer.rs
pub mod complete_transformer;


pub use config::TransformerConfig;
pub use transformer::Transformer;
pub use model_args::ModelArgs;
pub use metrics::MetricsTracker;
pub use generation::{TopKSampler, BeamSearch, NucleusSampler};
pub use monitoring::ExperimentMonitor;
pub use bleu::BleuScore;
pub use checkpoint::{CheckpointMetadata, save_checkpoint, load_checkpoint, find_latest_checkpoint, cleanup_old_checkpoints};
pub use inference::{generate_greedy, generate_beam_search, generate_sample, decode_tokens, encode_text};
