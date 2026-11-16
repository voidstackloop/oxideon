//! Neural Jump-Diffusion Stochastic Differential Equations (Jump-SDE) & Continuous Normalizing Flows (CNF).
//!
//! Features:
//! - Itô-Lévy stochastic jump-diffusion processes:
//!   $$d\mathbf{x}_t = f(\mathbf{x}_t, t) dt + g(\mathbf{x}_t, t) d\mathbf{W}_t + h(\mathbf{x}_t, t) d\mathbf{N}_t$$
//!   where $d\mathbf{W}_t \sim \mathcal{N}(0, dt \cdot I)$ is Brownian Wiener increments,
//!   and $d\mathbf{N}_t \sim \operatorname{Poisson}(\lambda dt)$ governs discontinuous compound jump events.
//! - Differentiable Euler-Maruyama stochastic integrator with Poisson jump increments.
//! - Continuous Normalizing Flow (CNF) instantaneous change of variables with Hutchinson's trace estimator:
//!   $$\frac{d \log p(\mathbf{x}_t)}{dt} = -\operatorname{Tr}\left(\frac{\partial f}{\partial \mathbf{x}_t}\right) \approx -\epsilon^T \left(\frac{\partial f}{\partial \mathbf{x}_t} \epsilon\right), \quad \epsilon \sim \mathcal{N}(0, I)$$

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::rng;
use crate::tensor::Tensor;

/// Jump-Diffusion Stochastic Simulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JumpDiffusionEngine {
    pub drift_scale: f64,
    pub diffusion_scale: f64,
    pub jump_intensity: f64,
    pub jump_scale: f64,
}

impl Default for JumpDiffusionEngine {
    fn default() -> Self {
        Self {
            drift_scale: 1.0,
            diffusion_scale: 0.1,
            jump_intensity: 0.5,
            jump_scale: 1.0,
        }
    }
}

impl JumpDiffusionEngine {
    /// Creates a new `JumpDiffusionEngine`.
    pub fn new(
        drift_scale: f64,
        diffusion_scale: f64,
        jump_intensity: f64,
        jump_scale: f64,
    ) -> Self {
        Self {
            drift_scale,
            diffusion_scale: diffusion_scale.max(0.0),
            jump_intensity: jump_intensity.max(0.0),
            jump_scale,
        }
    }

    /// Single Euler-Maruyama step with compound Poisson jump increment:
    /// $$x_{t+1} = x_t + f(x_t) dt + g \sqrt{dt} \xi + \mathbf{1}_{\{U < \lambda dt\}} \cdot J$$
    pub fn step(&self, x: &Tensor, drift: &Tensor, dt: f64) -> Result<Tensor> {
        let det_drift = drift.mul_scalar(self.drift_scale * dt)?;
        let mut x_next = x.add(&det_drift)?;

        // Continuous Brownian Wiener diffusion: g * sqrt(dt) * xi
        if self.diffusion_scale > 0.0 && dt > 0.0 {
            let brownian_std = self.diffusion_scale * dt.sqrt();
            let xi = Tensor::randn(x.shape())?
                .to_dtype(x.dtype())?
                .mul_scalar(brownian_std)?;
            x_next = x_next.add(&xi)?;
        }

        // Stochastic Poisson jump process; the trigger is a uniform draw, so
        // identical states do not jump identically (a sin-hash of the state
        // sum did that: deterministic pseudo-jumps correlated with x).
        let raw_intensity = self.jump_intensity * dt;
        if raw_intensity.is_nan() {
            return Err(Error::Msg("jump_intensity * dt is NaN".into()));
        }
        let jump_prob = raw_intensity.clamp(0.0, 1.0);
        if jump_prob > 0.0 {
            let u: f64 = rng::with_rng(|r| r.next_f32() as f64);
            if u < jump_prob {
                let jump_noise = Tensor::randn(x.shape())?
                    .to_dtype(x.dtype())?
                    .mul_scalar(self.jump_scale)?;
                x_next = x_next.add(&jump_noise)?;
            }
        }

        Ok(x_next)
    }

    /// Euler-Maruyama step with a state-dependent diffusion tensor
    /// $g(x_t)$ instead of the scalar `diffusion_scale`:
    /// $$x_{t+1} = x_t + f(x_t) dt + g(x_t) \sqrt{dt}\, \xi + \mathrm{jumps}$$
    /// Used by [`NeuralJumpSDE`], whose `diffusion_net` would otherwise be
    /// dead parameters. `diffusion` broadcasts against `x` like any operand.
    pub fn step_full(
        &self,
        x: &Tensor,
        drift: &Tensor,
        diffusion: &Tensor,
        dt: f64,
    ) -> Result<Tensor> {
        let det_drift = drift.mul_scalar(self.drift_scale * dt)?;
        let mut x_next = x.add(&det_drift)?;

        if dt > 0.0 {
            let g = diffusion.mul_scalar(dt.sqrt())?;
            let xi = Tensor::randn(x.shape())?.to_dtype(x.dtype())?;
            x_next = x_next.add(&g.mul(&xi)?)?;
        }

        let raw_intensity = self.jump_intensity * dt;
        if raw_intensity.is_nan() {
            return Err(Error::Msg("jump_intensity * dt is NaN".into()));
        }
        let jump_prob = raw_intensity.clamp(0.0, 1.0);
        if jump_prob > 0.0 {
            let u: f64 = rng::with_rng(|r| r.next_f32() as f64);
            if u < jump_prob {
                let jump_noise = Tensor::randn(x.shape())?
                    .to_dtype(x.dtype())?
                    .mul_scalar(self.jump_scale)?;
                x_next = x_next.add(&jump_noise)?;
            }
        }

        Ok(x_next)
    }

    /// Simulates a jump-diffusion trajectory over `steps` intervals of `dt`.
    pub fn simulate_trajectory<F>(
        &self,
        drift_fn: F,
        x0: &Tensor,
        dt: f64,
        steps: usize,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let steps = crate::limits::check_iterations("steps", steps)?;
        let mut traj = Vec::with_capacity(steps + 1);
        let mut x = x0.clone();
        traj.push(x.clone());

        for _ in 0..steps {
            let drift = drift_fn(&x)?;
            x = self.step(&x, &drift, dt)?;
            traj.push(x.clone());
        }

        Tensor::stack(&traj, 0)
    }

    /// Computes Hutchinson's stochastic trace estimator for the vector field divergence:
    /// $$\operatorname{div}(f)(x) = \operatorname{Tr}(\nabla_x f) \approx \frac{1}{S} \sum_{s=1}^S \epsilon_s^T \left( \frac{f(x + \delta \epsilon_s) - f(x - \delta \epsilon_s)}{2 \delta} \right)$$
    pub fn hutchinson_divergence<F>(drift_fn: F, x: &Tensor, num_samples: usize) -> Result<f64>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        let delta = 1e-4;
        let mut total_div = 0.0;
        let samples = crate::limits::check_iterations("num_samples", num_samples.max(1))?;

        for _ in 0..samples {
            let eps = Tensor::randn(x.shape())?.to_dtype(x.dtype())?;
            let x_plus = x.add(&eps.mul_scalar(delta)?)?;
            let x_minus = x.sub(&eps.mul_scalar(delta)?)?;

            let f_plus = drift_fn(&x_plus)?;
            let f_minus = drift_fn(&x_minus)?;

            let df = f_plus.sub(&f_minus)?.mul_scalar(1.0 / (2.0 * delta))?;
            let dot = eps.mul(&df)?.sum()?.item::<f64>()?;
            total_div += dot;
        }

        Ok(total_div / (samples as f64))
    }
}

/// Neural Jump-Diffusion SDE Module.
pub struct NeuralJumpSDE {
    pub drift_net: Linear,
    pub diffusion_net: Linear,
    pub jump_engine: JumpDiffusionEngine,
}

impl NeuralJumpSDE {
    /// Creates a new `NeuralJumpSDE`.
    pub fn new(dim: usize, jump_engine: JumpDiffusionEngine) -> Result<Self> {
        let drift_net = Linear::new(dim, dim)?;
        let diffusion_net = Linear::new(dim, dim)?;
        Ok(Self {
            drift_net,
            diffusion_net,
            jump_engine,
        })
    }

    /// Single continuous-time step forward through neural jump-SDE dynamics:
    /// drift and diffusion both come from their nets (the diffusion net
    /// used to be dead parameters: `step` only takes a scalar scale).
    pub fn forward_step(&self, x: &Tensor, dt: f64) -> Result<Tensor> {
        let drift = self.drift_net.forward(x)?;
        let diffusion = self.diffusion_net.forward(x)?;
        self.jump_engine.step_full(x, &drift, &diffusion, dt)
    }

    /// Returns trainable parameters of the neural jump-SDE.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.drift_net.parameters();
        p.extend(self.diffusion_net.parameters());
        p
    }
}
