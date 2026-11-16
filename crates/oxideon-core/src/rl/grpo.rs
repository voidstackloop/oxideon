//! DeepSeek-R1 Group Relative Policy Optimization (`GRPO`, Shao et al. 2024 / DeepSeek-AI 2025).
//!
//! GRPO removes the critic/value network completely from RL fine-tuning. Instead:
//! 1. For each prompt $q$, sample a group of $G$ outputs $\{o_1, \dots, o_G\}$.
//! 2. Evaluate reward $r_i$ for each response (e.g., format and mathematical accuracy).
//! 3. Normalize rewards across the group: $A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + \epsilon}$.
//! 4. Optimize the policy with clipped surrogate objective and token-level KL divergence penalty.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for GRPO post-training.
#[derive(Debug, Clone)]
pub struct GRPOConfig {
    /// Number of candidate responses sampled per prompt (typical: 4 to 8).
    pub group_size: usize,
    /// Clipping parameter $\epsilon$ for PPO surrogate objective (typical: 0.2).
    pub clip_eps: f64,
    /// Weight $\beta$ for token-level KL divergence against reference policy (typical: 0.04).
    pub kl_beta: f64,
    /// Small constant $\epsilon$ for advantage standard deviation normalization.
    pub eps: f64,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            group_size: 4,
            clip_eps: 0.2,
            kl_beta: 0.04,
            eps: 1e-6,
        }
    }
}

/// Computes group-normalized advantages from raw scalar rewards:
/// $$A_i = \frac{r_i - \operatorname{mean}(\{r\})}{\operatorname{std}(\{r\}) + \epsilon}$$
///
/// A `group_size` of 0 forms no groups and yields an empty vec instead of
/// panicking inside `chunks` ([`GRPOTrainer::compute_step`] rejects it loudly).
pub fn compute_group_advantages(rewards: &[f64], group_size: usize, eps: f64) -> Vec<f64> {
    if group_size == 0 {
        return Vec::new();
    }
    let mut advantages = Vec::with_capacity(rewards.len());
    for chunk in rewards.chunks(group_size) {
        let g = chunk.len() as f64;
        let mean = chunk.iter().sum::<f64>() / g.max(1.0);
        let var = chunk.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / g.max(1.0);
        let std = var.sqrt() + eps;
        for &r in chunk {
            advantages.push((r - mean) / std);
        }
    }
    advantages
}

/// GRPO clipped surrogate loss with per-token reference model KL penalty:
/// $$\mathcal{L} = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min( \rho_{it} A_i, \operatorname{clip}(\rho_{it}, 1-\epsilon, 1+\epsilon) A_i ) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right)$$
/// where $\rho_{it} = \exp(\log \pi_\theta - \log \pi_{\text{old}})$ and $D_{\text{KL}} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$.
pub fn grpo_loss(
    policy_logps: &Tensor,
    old_logps: &Tensor,
    ref_logps: &Tensor,
    advantages: &Tensor,
    mask: Option<&Tensor>,
    clip_eps: f64,
    kl_beta: f64,
) -> Result<Tensor> {
    if policy_logps.shape() != old_logps.shape() || policy_logps.shape() != ref_logps.shape() {
        return Err(Error::Msg(format!(
            "grpo_loss shape mismatch: policy {:?}, old {:?}, ref {:?}",
            policy_logps.shape(),
            old_logps.shape(),
            ref_logps.shape()
        )));
    }

    // Ratio rho = exp(log pi_theta - log pi_old)
    let log_diff = policy_logps.sub(&old_logps.detach())?;
    let ratio = log_diff.exp()?;

    // Advantage broadcast [G, 1] over [G, T]
    let adv = if advantages.ndim() == 1 {
        advantages.unsqueeze(1)?
    } else {
        advantages.clone()
    };

    let surr1 = ratio.mul(&adv)?;
    let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)?.mul(&adv)?;

    let use_surr1 = surr1.lt(&surr2)?;
    let min_surr = Tensor::where_cond(&use_surr1, &surr1, &surr2)?;

    // KL divergence penalty D_KL = exp(log_ref - log_pi) - (log_ref - log_pi) - 1.
    // The log-ratio is clamped to [-80, 80]: beyond that the probability
    // ratio exceeds 1e34 and `exp` only produces inf/NaN (80 keeps f32 finite).
    let ref_diff = ref_logps.detach().sub(policy_logps)?.clamp(-80.0, 80.0)?;
    let ratio_ref = ref_diff.exp()?;
    let kl = ratio_ref.sub(&ref_diff)?.sub_scalar(1.0)?;

    // Objective = min_surr - kl_beta * kl
    let per_token_obj = min_surr.sub(&kl.mul_scalar(kl_beta)?)?;

    // Apply token mask if sequence lengths vary
    let masked_obj = match mask {
        Some(m) => per_token_obj.mul(m)?,
        None => per_token_obj,
    };

    // Maximize objective => minimize negative objective. With a mask, average
    // over unmasked tokens only: dividing by all tokens would dilute the
    // loss toward 0 on padded batches. An all-masked batch yields 0.
    match mask {
        Some(m) => {
            let total = masked_obj.sum()?;
            let count = m
                .sum()?
                .to_dtype(total.dtype())?
                .clamp(1.0, f64::INFINITY)?;
            total.div(&count)?.neg()
        }
        None => masked_obj.neg()?.mean(),
    }
}

/// Rule-based reward evaluators for reasoning models (DeepSeek-R1 style).
#[derive(Debug, Clone, Default)]
pub struct RuleReward;

impl RuleReward {
    /// Format reward: verifies presence and proper strict nesting of `<think>...</think><answer>...</answer>`.
    /// Returns 1.0 for valid formatting, 0.0 otherwise.
    pub fn format_reward(text: &str) -> f64 {
        let has_open_think = text.contains("<think>");
        let has_close_think = text.contains("</think>");
        let has_open_ans = text.contains("<answer>");
        let has_close_ans = text.contains("</answer>");

        if !has_open_think || !has_close_think || !has_open_ans || !has_close_ans {
            return 0.0;
        }

        let think_start = text.find("<think>").unwrap();
        let think_end = text.find("</think>").unwrap();
        let ans_start = text.find("<answer>").unwrap();
        let ans_end = text.find("</answer>").unwrap();

        if think_start < think_end && think_end < ans_start && ans_start < ans_end {
            1.0
        } else {
            0.0
        }
    }

    /// Extracts the response content between `<answer>` and `</answer>` tags.
    pub fn extract_answer(text: &str) -> Option<String> {
        let start = text.find("<answer>")? + "<answer>".len();
        let end = text.find("</answer>")?;
        if start <= end {
            Some(text[start..end].trim().to_string())
        } else {
            None
        }
    }

    /// Evaluates exact answer match against ground truth.
    pub fn accuracy_reward(predicted: &str, ground_truth: &str) -> f64 {
        match Self::extract_answer(predicted) {
            Some(ans) if ans.eq_ignore_ascii_case(ground_truth.trim()) => 1.0,
            _ => 0.0,
        }
    }
}

/// Statistics collected across a GRPO training step.
#[derive(Debug, Clone)]
pub struct GRPOStepStats {
    pub mean_reward: f64,
    pub format_reward_rate: f64,
    pub loss: f64,
    pub mean_advantage: f64,
}

/// Group Relative Policy Optimization post-training coordinator.
pub struct GRPOTrainer {
    pub config: GRPOConfig,
}

impl GRPOTrainer {
    pub fn new(config: GRPOConfig) -> Self {
        Self { config }
    }

    /// Executes a single optimization step from sampled policy outputs and reference outputs.
    pub fn compute_step(
        &self,
        policy_logps: &Tensor,
        old_logps: &Tensor,
        ref_logps: &Tensor,
        raw_rewards: &[f64],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, GRPOStepStats)> {
        if self.config.group_size == 0 {
            return Err(Error::Msg(
                "GRPOTrainer: group_size must be positive".into(),
            ));
        }
        let adv_vec =
            compute_group_advantages(raw_rewards, self.config.group_size, self.config.eps);
        let n = adv_vec.len();
        let mean_reward = raw_rewards.iter().sum::<f64>() / (raw_rewards.len().max(1) as f64);
        let mean_adv = adv_vec.iter().sum::<f64>() / (n.max(1) as f64);

        let adv_tensor = Tensor::from_vec(adv_vec, &[n])?
            .to_device(&policy_logps.device())?
            .to_dtype(policy_logps.dtype())?;

        let loss = grpo_loss(
            policy_logps,
            old_logps,
            ref_logps,
            &adv_tensor,
            mask,
            self.config.clip_eps,
            self.config.kl_beta,
        )?;

        let loss_val = loss.item::<f32>()? as f64;

        let stats = GRPOStepStats {
            mean_reward,
            format_reward_rate: raw_rewards.iter().filter(|&&r| r > 0.0).count() as f64
                / (raw_rewards.len().max(1) as f64),
            loss: loss_val,
            mean_advantage: mean_adv,
        };

        Ok((loss, stats))
    }
}
