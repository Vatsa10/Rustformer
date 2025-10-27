use nalgebra::DMatrix;
use std::collections::HashMap;

/// AdamW optimizer implementation with weight decay
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub step: usize,
    
    // First moment estimates (momentum)
    m: HashMap<String, DMatrix<f64>>,
    // Second moment estimates (RMSprop)
    v: HashMap<String, DMatrix<f64>>,
}

impl AdamW {
    pub fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
    
    /// Update parameters with AdamW algorithm
    pub fn step(&mut self, param_name: &str, params: &DMatrix<f64>, grads: &DMatrix<f64>) -> DMatrix<f64> {
        self.step += 1;
        
        // Initialize moment estimates if needed
        if !self.m.contains_key(param_name) {
            self.m.insert(param_name.to_string(), DMatrix::zeros(params.nrows(), params.ncols()));
            self.v.insert(param_name.to_string(), DMatrix::zeros(params.nrows(), params.ncols()));
        }
        
        let m = self.m.get_mut(param_name).unwrap();
        let v = self.v.get_mut(param_name).unwrap();
        
        // Update biased first moment estimate
        *m = m.clone() * self.beta1 + grads * (1.0 - self.beta1);
        
        // Update biased second raw moment estimate
        let grads_squared = grads.component_mul(grads);
        *v = v.clone() * self.beta2 + grads_squared * (1.0 - self.beta2);
        
        // Compute bias-corrected first moment estimate
        let m_hat = m.clone() / (1.0 - self.beta1.powi(self.step as i32));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = v.clone() / (1.0 - self.beta2.powi(self.step as i32));
        
        // Update parameters with weight decay (AdamW)
        let weight_decay_term = params * self.weight_decay * self.lr;
        let adaptive_lr = m_hat.component_div(&(v_hat.map(|x| x.sqrt()) + DMatrix::from_element(v_hat.nrows(), v_hat.ncols(), self.epsilon)));
        
        params - adaptive_lr * self.lr - weight_decay_term
    }
    
    /// Update learning rate
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    
    /// Reset optimizer state
    pub fn zero_grad(&mut self) {
        self.m.clear();
        self.v.clear();
    }
}

/// Gradient clipping by global norm
pub fn clip_gradients(gradients: &mut Vec<DMatrix<f64>>, max_norm: f64) -> f64 {
    // Compute global norm
    let mut total_norm = 0.0;
    for grad in gradients.iter() {
        total_norm += grad.iter().map(|x| x * x).sum::<f64>();
    }
    total_norm = total_norm.sqrt();
    
    // Clip if necessary
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for grad in gradients.iter_mut() {
            *grad = grad.clone() * clip_coef;
        }
    }
    
    total_norm
}