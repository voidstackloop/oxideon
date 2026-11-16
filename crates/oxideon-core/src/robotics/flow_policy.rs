//! Continuous Flow Matching Robotics Policy.
//!
//! Generates smooth end-effector trajectories and joint actions by solving the probability
//! flow ODE: $d a_\tau / d\tau = v_\theta(a_\tau, \tau, \text{obs})$ from standard Gaussian noise.

use crate::error::Result;
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Flow Matching Continuous Robotics Policy.
pub struct FlowMatchingRoboticsPolicy {
    pub action_dim: usize,
    pub obs_dim: usize,
    pub net1: Linear,
    pub net2: Linear,
    pub out_net: Linear,
}

impl FlowMatchingRoboticsPolicy {
    pub fn new(action_dim: usize, obs_dim: usize, hidden_dim: usize) -> Result<Self> {
        // Input: [action (action_dim), time tau (1), observation (obs_dim)]
        let in_dim = action_dim
            .checked_add(1)
            .and_then(|v| v.checked_add(obs_dim))
            .ok_or_else(|| {
                crate::error::Error::Msg(format!(
                    "FlowMatchingRoboticsPolicy: input dim {action_dim} + 1 + {obs_dim} overflows"
                ))
            })?;
        let net1 = Linear::new(in_dim, hidden_dim)?;
        let net2 = Linear::new(hidden_dim, hidden_dim)?;
        let out_net = Linear::new(hidden_dim, action_dim)?;

        Ok(Self {
            action_dim,
            obs_dim,
            net1,
            net2,
            out_net,
        })
    }

    /// Evaluates vector velocity field $v_\theta(a_\tau, \tau, \text{obs})$.
    pub fn forward(&self, a_tau: &Tensor, tau: &Tensor, obs: &Tensor) -> Result<Tensor> {
        let input = Tensor::cat(&[a_tau.clone(), tau.clone(), obs.clone()], 1)?;
        let h = self.net1.forward(&input)?.silu()?;
        let h2 = self.net2.forward(&h)?.silu()?;
        self.out_net.forward(&h2)
    }

    /// Generates action trajectory using Euler ODE integration over `num_steps`.
    pub fn sample_action(&self, obs: &Tensor, num_steps: usize) -> Result<Tensor> {
        if obs.ndim() < 1 {
            return Err(crate::error::Error::Msg(format!(
                "sample_action needs [batch, obs_dim] observation, got {:?}",
                obs.shape()
            )));
        }
        // Each step runs full network passes; an uncapped count never returns.
        let num_steps = crate::limits::check_iterations("num_steps", num_steps)?;
        let b = obs.shape()[0];
        let mut a = Tensor::randn(&[b, self.action_dim])?
            .to_device(&obs.device())?
            .to_dtype(obs.dtype())?;
        let dt = 1.0 / (num_steps.max(1) as f64);

        for step in 0..num_steps {
            let tau_val = (step as f64) * dt;
            let tau = Tensor::from_vec(vec![tau_val as f32; b], &[b, 1])?
                .to_device(&obs.device())?
                .to_dtype(obs.dtype())?;

            let v = self.forward(&a, &tau, obs)?;
            a = a.add(&v.mul_scalar(dt)?)?;
        }

        Ok(a)
    }
}
