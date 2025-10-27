use serde_json::json;
use std::collections::HashMap;
use chrono::Utc;

/// WandB-like monitoring system for experiment tracking
pub struct ExperimentMonitor {
    pub project_name: String,
    pub run_name: String,
    pub config: HashMap<String, serde_json::Value>,
    pub metrics: HashMap<String, Vec<(usize, f64)>>,
    pub start_time: chrono::DateTime<Utc>,
}

impl ExperimentMonitor {
    pub fn new(project_name: &str, run_name: &str) -> Self {
        Self {
            project_name: project_name.to_string(),
            run_name: run_name.to_string(),
            config: HashMap::new(),
            metrics: HashMap::new(),
            start_time: Utc::now(),
        }
    }
    
    /// Log configuration
    pub fn log_config<T: serde::Serialize>(&mut self, config: &T) {
        if let Ok(json_val) = serde_json::to_value(config) {
            if let Some(obj) = json_val.as_object() {
                for (k, v) in obj {
                    self.config.insert(k.clone(), v.clone());
                }
            }
        }
    }
    
    /// Log a metric at a specific step
    pub fn log_metric(&mut self, name: &str, value: f64, step: usize) {
        self.metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push((step, value));
    }
    
    /// Log multiple metrics at once
    pub fn log_metrics(&mut self, metrics: HashMap<String, f64>, step: usize) {
        for (name, value) in metrics {
            self.log_metric(&name, value, step);
        }
    }
    
    /// Print current metrics
    pub fn print_metrics(&self, step: usize) {
        println!("\n📈 Experiment: {} / {}", self.project_name, self.run_name);
        println!("Step {}: ", step);
        
        // Print recent metrics
        for (name, values) in &self.metrics {
            if let Some((_, value)) = values.last() {
                println!("  • {}: {:.6}", name, value);
            }
        }
    }
    
    /// Log generation sample
    pub fn log_generation(&self, step: usize, prompt: &str, generated: &str) {
        println!("\n🎯 Generation Sample (Step {}):", step);
        println!("  Prompt: {}", prompt);
        println!("  Generated: {}", generated);
    }
    
    /// Save metrics to JSON file
    pub fn save_to_json(&self, filename: &str) -> anyhow::Result<()> {
        let data = json!({
            "project": self.project_name,
            "run": self.run_name,
            "config": self.config,
            "metrics": self.metrics,
            "start_time": self.start_time.to_rfc3339(),
            "end_time": Utc::now().to_rfc3339(),
        });
        
        let json_str = serde_json::to_string_pretty(&data)?;
        std::fs::write(filename, json_str)?;
        
        println!("💾 Metrics saved to: {}", filename);
        Ok(())
    }
    
    /// Get summary statistics
    pub fn get_summary(&self) -> String {
        let duration = Utc::now() - self.start_time;
        format!(
            "Experiment: {}/{}\nDuration: {} seconds\nMetrics tracked: {}",
            self.project_name,
            self.run_name,
            duration.num_seconds(),
            self.metrics.len()
        )
    }
}