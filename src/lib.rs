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

// This file is intentionally left blank as its content has been merged into transformer.rs
pub mod complete_transformer;


pub use config::TransformerConfig;
pub use transformer::Transformer;
pub use model_args::ModelArgs;
pub use metrics::MetricsTracker;
pub use generation::{TopKSampler, BeamSearch, NucleusSampler};
pub use monitoring::ExperimentMonitor;
