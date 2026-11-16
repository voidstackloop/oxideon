//! Direct Preference Optimization (`DPO`, Rafailov et al. 2023).
//!
//! Replaces explicit reward modeling and reinforcement learning with an exact closed-form
//! implicit reward objective over pairwise human preference data $(x, y_w, y_l)$.

use crate::error::Result;
use crate::tensor::Tensor;

/// Configuration for DPO post-training.
#[derive(Debug, Clone)]
pub struct DPOConfig {
    /// Temperature parameter $\beta$ controlling reference model divergence penalty (typical: 0.1 to 0.5).
    pub beta: f64,
    /// Label smoothing factor $\epsilon$ (typical: 0.0 to 0.1).
    pub label_smoothing: f64,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
        }
    }
}

/// Statistics reported after a DPO forward step.
#[derive(Debug, Clone)]
pub struct DPOStepStats {
    pub loss: f64,
    pub chosen_reward: f64,
    pub rejected_reward: f64,
    pub reward_margin: f64,
    pub accuracy: f64,
}

/// DPO preference optimization trainer.
pub struct DPOTrainer {
    pub config: DPOConfig,
}

impl DPOTrainer {
    pub fn new(config: DPOConfig) -> Self {
        Self { config }
    }

    /// Computes DPO loss and tracking metrics from sequence log-probabilities.
    ///
    /// - `policy_chosen_logps`: $\log \pi_\theta(y_w \mid x)$ of shape `[batch]`
    /// - `policy_rejected_logps`: $\log \pi_\theta(y_l \mid x)$ of shape `[batch]`
    /// - `ref_chosen_logps`: $\log \pi_{\text{ref}}(y_w \mid x)$ of shape `[batch]`
    /// - `ref_rejected_logps`: $\log \pi_{\text{ref}}(y_l \mid x)$ of shape `[batch]`
    pub fn compute_step(
        &self,
        policy_chosen_logps: &Tensor,
        policy_rejected_logps: &Tensor,
        ref_chosen_logps: &Tensor,
        ref_rejected_logps: &Tensor,
    ) -> Result<(Tensor, DPOStepStats)> {
        let pi_chosen = policy_chosen_logps;
        let pi_rejected = policy_rejected_logps;
        let ref_chosen = ref_chosen_logps.detach();
        let ref_rejected = ref_rejected_logps.detach();

        // Implicit rewards: r = beta * (log pi - log ref)
        let chosen_logratios = pi_chosen.sub(&ref_chosen)?;
        let rejected_logratios = pi_rejected.sub(&ref_rejected)?;

        let chosen_rewards = chosen_logratios.mul_scalar(self.config.beta)?;
        let rejected_rewards = rejected_logratios.mul_scalar(self.config.beta)?;

        let logits = chosen_rewards.sub(&rejected_rewards)?;

        // DPO loss: -log(sigmoid(logits)) = softplus(-logits)
        let loss = if self.config.label_smoothing > 0.0 {
            let ls = self.config.label_smoothing;
            let term1 = logits.neg()?.softplus()?.mul_scalar(1.0 - ls)?;
            let term2 = logits.softplus()?.mul_scalar(ls)?;
            term1.add(&term2)?.mean()?
        } else {
            logits.neg()?.softplus()?.mean()?
        };

        // Extract statistics
        let r_w_vec = chosen_rewards.to_vec::<f32>()?;
        let r_l_vec = rejected_rewards.to_vec::<f32>()?;
        let n = r_w_vec.len().max(1);

        let mean_r_w: f64 = r_w_vec.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);
        let mean_r_l: f64 = r_l_vec.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);
        let margin = mean_r_w - mean_r_l;

        let acc_count = r_w_vec
            .iter()
            .zip(r_l_vec.iter())
            .filter(|&(w, l)| *w > *l)
            .count();
        let accuracy = (acc_count as f64) / (n as f64);
        let loss_val = loss.item::<f32>()? as f64;

        let stats = DPOStepStats {
            loss: loss_val,
            chosen_reward: mean_r_w,
            rejected_reward: mean_r_l,
            reward_margin: margin,
            accuracy,
        };

        Ok((loss, stats))
    }
}
