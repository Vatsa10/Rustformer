use std::collections::VecDeque;

/// Metrics tracker for monitoring training
pub struct MetricsTracker {
    pub train_losses: VecDeque<f64>,
    pub val_losses: VecDeque<f64>,
    pub learning_rates: VecDeque<f64>,
    pub gradient_norms: VecDeque<f64>,
    pub perplexities: VecDeque<f64>,
    max_history: usize,
}

impl MetricsTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            train_losses: VecDeque::new(),
            val_losses: VecDeque::new(),
            learning_rates: VecDeque::new(),
            gradient_norms: VecDeque::new(),
            perplexities: VecDeque::new(),
            max_history,
        }
    }
    
    /// Log training loss
    pub fn log_train_loss(&mut self, loss: f64) {
        self.train_losses.push_back(loss);
        if self.train_losses.len() > self.max_history {
            self.train_losses.pop_front();
        }
    }
    
    /// Log validation loss
    pub fn log_val_loss(&mut self, loss: f64) {
        self.val_losses.push_back(loss);
        if self.val_losses.len() > self.max_history {
            self.val_losses.pop_front();
        }
    }
    
    /// Log learning rate
    pub fn log_lr(&mut self, lr: f64) {
        self.learning_rates.push_back(lr);
        if self.learning_rates.len() > self.max_history {
            self.learning_rates.pop_front();
        }
    }
    
    /// Log gradient norm
    pub fn log_gradient_norm(&mut self, norm: f64) {
        self.gradient_norms.push_back(norm);
        if self.gradient_norms.len() > self.max_history {
            self.gradient_norms.pop_front();
        }
    }
    
    /// Compute and log perplexity from loss
    pub fn log_perplexity(&mut self, loss: f64) {
        let perplexity = loss.exp();
        self.perplexities.push_back(perplexity);
        if self.perplexities.len() > self.max_history {
            self.perplexities.pop_front();
        }
    }
    
    /// Get average of recent training losses
    pub fn avg_train_loss(&self, last_n: usize) -> f64 {
        let n = last_n.min(self.train_losses.len());
        if n == 0 {
            return 0.0;
        }
        self.train_losses.iter().rev().take(n).sum::<f64>() / n as f64
    }
    
    /// Get average of recent validation losses
    pub fn avg_val_loss(&self, last_n: usize) -> f64 {
        let n = last_n.min(self.val_losses.len());
        if n == 0 {
            return 0.0;
        }
        self.val_losses.iter().rev().take(n).sum::<f64>() / n as f64
    }
    
    /// Get current perplexity
    pub fn current_perplexity(&self) -> f64 {
        self.perplexities.back().copied().unwrap_or(0.0)
    }
    
    /// Get current gradient norm
    pub fn current_gradient_norm(&self) -> f64 {
        self.gradient_norms.back().copied().unwrap_or(0.0)
    }
    
    /// Get current learning rate
    pub fn current_lr(&self) -> f64 {
        self.learning_rates.back().copied().unwrap_or(0.0)
    }
    
    /// Print summary statistics
    pub fn print_summary(&self, step: usize) {
        println!("\n📊 Training Metrics Summary (Step {}):", step);
        println!("  • Avg Train Loss (last 100): {:.4}", self.avg_train_loss(100));
        println!("  • Avg Val Loss: {:.4}", self.avg_val_loss(10));
        println!("  • Current Perplexity: {:.2}", self.current_perplexity());
        println!("  • Current Gradient Norm: {:.4}", self.current_gradient_norm());
        println!("  • Current Learning Rate: {:.2e}", self.current_lr());
    }
}