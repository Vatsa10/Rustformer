use anyhow::Result;
use candle_core::Device;
use candle_nn::VarMap;
use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};

use crate::ModelArgs;

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub model_args: ModelArgs,
    pub timestamp: String,
}

/// Save model checkpoint
pub fn save_checkpoint(
    varmap: &VarMap,
    metadata: CheckpointMetadata,
    checkpoint_dir: &str,
) -> Result<()> {
    // Create checkpoint directory if it doesn't exist
    fs::create_dir_all(checkpoint_dir)?;
    
    let checkpoint_name = format!("checkpoint_step_{}.safetensors", metadata.step);
    let checkpoint_path = Path::new(checkpoint_dir).join(&checkpoint_name);
    
    // Save model weights
    varmap.save(&checkpoint_path)?;
    
    // Save metadata
    let metadata_path = Path::new(checkpoint_dir).join(format!("checkpoint_step_{}.json", metadata.step));
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(metadata_path, metadata_json)?;
    
    println!("💾 Checkpoint saved: {}", checkpoint_path.display());
    
    Ok(())
}

/// Load model checkpoint
pub fn load_checkpoint(
    varmap: &mut VarMap,
    checkpoint_path: &str,
    _device: &Device,
) -> Result<CheckpointMetadata> {
    // Load model weights
    varmap.load(checkpoint_path)?;
    
    // Load metadata
    let metadata_path = checkpoint_path.replace(".safetensors", ".json");
    let metadata_json = fs::read_to_string(&metadata_path)?;
    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;
    
    println!("📂 Checkpoint loaded: {}", checkpoint_path);
    println!("   Step: {}, Loss: {:.4}", metadata.step, metadata.loss);
    
    Ok(metadata)
}

/// Find the latest checkpoint in a directory
pub fn find_latest_checkpoint(checkpoint_dir: &str) -> Result<Option<String>> {
    if !Path::new(checkpoint_dir).exists() {
        return Ok(None);
    }
    
    let mut checkpoints = Vec::new();
    
    for entry in fs::read_dir(checkpoint_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                if let Some(filename) = path.file_name() {
                    if let Some(filename_str) = filename.to_str() {
                        // Extract step number from filename
                        if let Some(step_str) = filename_str
                            .strip_prefix("checkpoint_step_")
                            .and_then(|s| s.strip_suffix(".safetensors"))
                        {
                            if let Ok(step) = step_str.parse::<usize>() {
                                checkpoints.push((step, path.to_string_lossy().to_string()));
                            }
                        }
                    }
                }
            }
        }
    }
    
    if checkpoints.is_empty() {
        return Ok(None);
    }
    
    // Sort by step number and return the latest
    checkpoints.sort_by_key(|(step, _)| *step);
    Ok(Some(checkpoints.last().unwrap().1.clone()))
}

/// Clean up old checkpoints, keeping only the last N
pub fn cleanup_old_checkpoints(checkpoint_dir: &str, keep_last_n: usize) -> Result<()> {
    if !Path::new(checkpoint_dir).exists() {
        return Ok(());
    }
    
    let mut checkpoints = Vec::new();
    
    for entry in fs::read_dir(checkpoint_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                if let Some(filename) = path.file_name() {
                    if let Some(filename_str) = filename.to_str() {
                        if let Some(step_str) = filename_str
                            .strip_prefix("checkpoint_step_")
                            .and_then(|s| s.strip_suffix(".safetensors"))
                        {
                            if let Ok(step) = step_str.parse::<usize>() {
                                checkpoints.push((step, path.clone()));
                            }
                        }
                    }
                }
            }
        }
    }
    
    if checkpoints.len() <= keep_last_n {
        return Ok(());
    }
    
    // Sort by step number
    checkpoints.sort_by_key(|(step, _)| *step);
    
    // Remove old checkpoints
    let to_remove = checkpoints.len() - keep_last_n;
    for (_, path) in checkpoints.iter().take(to_remove) {
        // Remove both .safetensors and .json files
        fs::remove_file(path)?;
        let json_path = path.to_string_lossy().replace(".safetensors", ".json");
        if Path::new(&json_path).exists() {
            fs::remove_file(json_path)?;
        }
        println!("🗑️  Removed old checkpoint: {}", path.display());
    }
    
    Ok(())
}
