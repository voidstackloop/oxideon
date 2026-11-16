//! Early-Exit Adaptive Computation Router for Dynamic Depth Inference.
//!
//! Evaluates intermediate prediction confidence (via maximum softmax probability or negative entropy)
//! allowing easy tokens to exit early from the model without computing deeper layers.

use crate::error::Result;
use crate::tensor::Tensor;

/// Early-exit decision router.
#[derive(Debug, Clone)]
pub struct EarlyExitRouter {
    pub confidence_threshold: f64,
}

impl EarlyExitRouter {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Evaluates logits `[batch, vocab_size]` and returns `(exit_flags, max_probs)`.
    pub fn evaluate_exit(&self, logits: &Tensor) -> Result<(Vec<bool>, Tensor)> {
        let probs = logits.softmax(-1)?;
        let max_prob = probs.max_axis(-1, false)?;
        let max_prob_vec = max_prob.to_vec::<f32>()?;

        let flags: Vec<bool> = max_prob_vec
            .iter()
            .map(|&p| (p as f64) >= self.confidence_threshold)
            .collect();

        Ok((flags, max_prob))
    }
}
