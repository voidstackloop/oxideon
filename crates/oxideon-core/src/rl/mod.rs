//! Reinforcement Learning and Post-Training Alignment Engine.
//!
//! Includes:
//! - DeepSeek-R1 Group Relative Policy Optimization ([`GRPOTrainer`], [`grpo_loss`], [`compute_group_advantages`], [`RuleReward`])
//! - Direct Preference Optimization ([`DPOTrainer`], [`DPOConfig`])
//! - Proximal Policy Optimization Actor-Critic ([`PPOTrainer`], [`PPOConfig`])

pub mod classic;
pub mod dpo;
pub mod grpo;
pub mod ppo;

pub use dpo::{DPOConfig, DPOStepStats, DPOTrainer};
pub use grpo::{
    GRPOConfig, GRPOStepStats, GRPOTrainer, RuleReward, compute_group_advantages, grpo_loss,
};
pub use ppo::{PPOConfig, PPOStepStats, PPOTrainer};
