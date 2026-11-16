//! Proximal Policy Optimization (`PPO`, Schulman et al. 2017) Actor-Critic.
//!
//! Features clipped surrogate policy objective, clipped value function loss, and entropy bonus.

use crate::error::Result;
use crate::nn::loss::ppo_clipped_loss;
use crate::tensor::Tensor;

/// Configuration for PPO actor-critic training.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Policy ratio clipping range $\epsilon$ (typical: 0.1 to 0.2).
    pub clip_eps: f64,
    /// Value function loss weight $c_1$ (typical: 0.5 to 1.0).
    pub vf_coef: f64,
    /// Entropy bonus weight $c_2$ (typical: 0.01).
    pub ent_coef: f64,
    /// Value clipping parameter $\epsilon_v$ (optional).
    pub clip_vf: Option<f64>,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            clip_vf: Some(0.2),
        }
    }
}

/// Statistics from a PPO actor-critic training step.
#[derive(Debug, Clone)]
pub struct PPOStepStats {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub total_loss: f64,
}

/// PPO Actor-Critic optimization manager.
pub struct PPOTrainer {
    pub config: PPOConfig,
}

impl PPOTrainer {
    pub fn new(config: PPOConfig) -> Self {
        Self { config }
    }

    /// Computes total PPO objective:
    /// $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}} - c_2 \mathcal{S}_{\text{entropy}}$$
    pub fn compute_step(
        &self,
        new_logps: &Tensor,
        old_logps: &Tensor,
        advantages: &Tensor,
        new_values: &Tensor,
        old_values: &Tensor,
        returns: &Tensor,
    ) -> Result<(Tensor, PPOStepStats)> {
        // 1. Clipped Policy Loss
        let policy_loss = ppo_clipped_loss(new_logps, old_logps, advantages, self.config.clip_eps)?;

        // 2. Value Function Loss (with optional clipping)
        let v_diff = new_values.sub(returns)?;
        let vf_unclipped = v_diff.powf(2.0)?;

        let value_loss = match self.config.clip_vf {
            Some(eps_v) => {
                let v_clipped = old_values
                    .detach()
                    .add(&new_values.sub(&old_values.detach())?.clamp(-eps_v, eps_v)?)?;
                let vf_clipped = v_clipped.sub(returns)?.powf(2.0)?;
                let use_unclipped = vf_unclipped.gt(&vf_clipped)?;
                Tensor::where_cond(&use_unclipped, &vf_unclipped, &vf_clipped)?.mean()?
            }
            None => vf_unclipped.mean()?,
        };

        // 3. Approximate Entropy Bonus: -mean(log_p)
        let entropy = new_logps.neg()?.mean()?;

        // Combined Loss = L_policy + c1 * L_value - c2 * Entropy
        let total_loss = policy_loss
            .add(&value_loss.mul_scalar(self.config.vf_coef)?)?
            .sub(&entropy.mul_scalar(self.config.ent_coef)?)?;

        let pol_val = policy_loss.item::<f32>()? as f64;
        let val_val = value_loss.item::<f32>()? as f64;
        let ent_val = entropy.item::<f32>()? as f64;
        let tot_val = total_loss.item::<f32>()? as f64;

        let stats = PPOStepStats {
            policy_loss: pol_val,
            value_loss: val_val,
            entropy: ent_val,
            total_loss: tot_val,
        };

        Ok((total_loss, stats))
    }
}
