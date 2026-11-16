//! Unified 4D World Model & Video-Action Diffusion Engine.
//!
//! Integrates continuous spatio-temporal video diffusion with embodied robotic action
//! trajectory generation. Enables dual-mode prediction:
//! 1. Forward World Simulation: predicts future video dynamics conditioned on planned actions.
//! 2. Inverse Goal-Conditioned Policy: synthesizes robotic control actions to achieve visual goals.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Configuration for Unified 4D World Model.
#[derive(Debug, Clone)]
pub struct WorldModelConfig {
    pub latent_channels: usize,
    pub action_dim: usize,
    pub action_chunk: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            latent_channels: 4,
            action_dim: 7,
            action_chunk: 8,
            hidden_dim: 64,
            num_layers: 2,
        }
    }
}

/// Unified 4D Generative World Model and Robotics Policy.
pub struct UnifiedWorldModel {
    pub config: WorldModelConfig,
    pub visual_proj: Linear,
    pub action_proj: Linear,
    pub time_proj: Linear,
    pub joint_blocks: Vec<Linear>,
    pub video_head: Linear,
    pub action_head: Linear,
}

impl UnifiedWorldModel {
    pub fn new(config: WorldModelConfig) -> Result<Self> {
        let visual_proj = Linear::new(config.latent_channels, config.hidden_dim)?;
        let action_proj = Linear::new(config.action_dim, config.hidden_dim)?;
        let time_proj = Linear::new(1, config.hidden_dim)?;

        crate::limits::check_repeats("num_layers", config.num_layers)?;
        let mut joint_blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            joint_blocks.push(Linear::new(config.hidden_dim, config.hidden_dim)?);
        }

        let video_head = Linear::new(config.hidden_dim, config.latent_channels)?;
        let action_out = config
            .action_dim
            .checked_mul(config.action_chunk)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "UnifiedWorldModel: action dim {} * {} overflows",
                    config.action_dim, config.action_chunk
                ))
            })?;
        let action_head = Linear::new(config.hidden_dim, action_out)?;

        Ok(Self {
            config,
            visual_proj,
            action_proj,
            time_proj,
            joint_blocks,
            video_head,
            action_head,
        })
    }

    /// Evaluates forward dynamics: predicts visual flow field conditioned on action trajectory.
    /// `latents`: `[batch, seq_tokens, channels]`
    /// `actions`: `[batch, action_dim]`
    pub fn forward_dynamics(&self, latents: &Tensor, actions: &Tensor, tau: f64) -> Result<Tensor> {
        let b = latents.shape()[0];
        let v_tok = self.visual_proj.forward(latents)?;
        let a_tok = self.action_proj.forward(actions)?.unsqueeze(1)?;

        let tau_t = Tensor::from_vec(vec![tau as f32; b], &[b, 1])?
            .to_device(&latents.device())?
            .to_dtype(latents.dtype())?;
        let t_tok = self.time_proj.forward(&tau_t)?.unsqueeze(1)?;

        let mut h = v_tok.add(&a_tok)?.add(&t_tok)?;
        for block in &self.joint_blocks {
            let res = h.clone();
            h = block.forward(&h)?.silu()?.add(&res)?;
        }

        self.video_head.forward(&h)
    }

    /// Generates future video latents given initial state and action sequence via Euler ODE integration.
    pub fn sample_future_video(
        &self,
        init_latents: &Tensor,
        actions: &Tensor,
        num_steps: usize,
    ) -> Result<Tensor> {
        let mut v = init_latents.clone();
        let dt = 1.0 / (num_steps.max(1) as f64);

        for step in 0..num_steps {
            let tau = (step as f64) * dt;
            let flow = self.forward_dynamics(&v, actions, tau)?;
            v = v.add(&flow.mul_scalar(dt)?)?;
        }

        Ok(v)
    }

    /// Synthesizes robot action chunk `[batch, action_chunk, action_dim]` given current and goal latents.
    pub fn plan_actions_to_goal(
        &self,
        curr_latents: &Tensor,
        goal_latents: &Tensor,
    ) -> Result<Tensor> {
        let b = curr_latents.shape()[0];
        let diff = goal_latents.sub(curr_latents)?;
        let h_diff = self.visual_proj.forward(&diff)?.mean_axis(-2, false)?;

        let mut h = h_diff;
        for block in &self.joint_blocks {
            let res = h.clone();
            h = block.forward(&h)?.silu()?.add(&res)?;
        }

        let flat_actions = self.action_head.forward(&h)?;
        flat_actions.reshape(&[b, self.config.action_chunk, self.config.action_dim])
    }
}
