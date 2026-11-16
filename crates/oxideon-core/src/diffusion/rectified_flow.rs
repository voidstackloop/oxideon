//! Rectified Flow Engine & Fast ODE Sampler (Flux.1 / SD3 / InstaFlow).
//!
//! Provides:
//! - Straight trajectory sampling: $x_t = (1 - t) x_0 + t x_1$ with velocity target $v = x_1 - x_0$.
//! - Classifier-Free Guidance (CFG): $v_{\text{eff}} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$.
//! - 2-Rectified Flow (Reflow) dataset generation for distilling diffusion models down to 1-4 steps.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for Rectified Flow sampling.
#[derive(Debug, Clone)]
pub struct RectifiedFlowConfig {
    pub num_steps: usize,
    pub guidance_scale: f64,
}

impl Default for RectifiedFlowConfig {
    fn default() -> Self {
        Self {
            num_steps: 20,
            guidance_scale: 1.0,
        }
    }
}

/// Rectified Flow execution and distillation engine.
pub struct RectifiedFlowEngine;

impl RectifiedFlowEngine {
    /// Linearly interpolates prior noise `x0` and data `x1` at time `t` $\in [0, 1]$.
    ///
    /// $$x_t = (1 - t) x_0 + t x_1$$
    /// Velocity ground-truth target:
    /// $$v_t = x_1 - x_0$$
    pub fn interpolate(x0: &Tensor, x1: &Tensor, t: f64) -> Result<(Tensor, Tensor)> {
        if x0.shape() != x1.shape() {
            return Err(Error::ShapeMismatch {
                op: "RectifiedFlowEngine::interpolate",
                expected: x0.shape().to_vec(),
                got: x1.shape().to_vec(),
            });
        }
        let t_clamped = t.clamp(0.0, 1.0);
        let part_0 = x0.mul_scalar(1.0 - t_clamped)?;
        let part_1 = x1.mul_scalar(t_clamped)?;
        let xt = part_0.add(&part_1)?;
        let target_v = x1.sub(x0)?;
        Ok((xt, target_v))
    }

    /// Generates samples from initial noise `x0` using high-order Euler ODE integration
    /// with optional Classifier-Free Guidance (CFG).
    ///
    /// `velocity_fn`: `Fn(x: &Tensor, t: f64, cond: bool) -> Result<Tensor>`.
    pub fn sample_euler<F>(
        x0: &Tensor,
        config: &RectifiedFlowConfig,
        velocity_fn: F,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, f64, bool) -> Result<Tensor>,
    {
        let steps =
            crate::limits::check_iterations("rectified_flow num_steps", config.num_steps.max(1))?;
        let dt = 1.0 / (steps as f64);
        let mut x = x0.clone();

        for i in 0..steps {
            let t = (i as f64) * dt;

            let v_eff = if (config.guidance_scale - 1.0).abs() < 1e-6 {
                // Unconditional / single-pass
                velocity_fn(&x, t, true)?
            } else {
                // Classifier-Free Guidance: v_uncond + s * (v_cond - v_uncond)
                let v_cond = velocity_fn(&x, t, true)?;
                let v_uncond = velocity_fn(&x, t, false)?;
                let diff = v_cond.sub(&v_uncond)?;
                let guided = diff.mul_scalar(config.guidance_scale)?;
                v_uncond.add(&guided)?
            };

            // Euler step: x_{t+dt} = x_t + dt * v_eff
            let delta = v_eff.mul_scalar(dt)?;
            x = x.add(&delta)?;
        }

        Ok(x)
    }

    /// Performs Reflow trajectory pairs generation for 2-Rectified Flow distillation:
    /// Takes noise samples `x0` and solves ODE to find predicted endpoints `x1`,
    /// returning straight pairs `(x0, x1)` for one-step / few-step reflow training.
    pub fn generate_reflow_pairs<F>(
        x0_batch: &[Tensor],
        steps: usize,
        velocity_fn: F,
    ) -> Result<Vec<(Tensor, Tensor)>>
    where
        F: Fn(&Tensor, f64, bool) -> Result<Tensor>,
    {
        let cfg = RectifiedFlowConfig {
            num_steps: steps,
            guidance_scale: 1.0,
        };
        let mut pairs = Vec::with_capacity(x0_batch.len());
        for x0 in x0_batch {
            let x1 = Self::sample_euler(x0, &cfg, &velocity_fn)?;
            pairs.push((x0.clone(), x1));
        }
        Ok(pairs)
    }
}
