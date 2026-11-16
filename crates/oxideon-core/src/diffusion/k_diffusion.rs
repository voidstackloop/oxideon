//! K-Diffusion & EDM2 High-Order Continuous-Time Diffusion Solvers (Karras et al.).
//!
//! Provides continuous variance-exploding and variance-preserving noise schedules:
//!
//! $$\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N - 1} (\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}) \right)^\rho$$
//!
//! alongside Heun's 2nd-order predictor-corrector ODE sampler for ultra-fast high-fidelity sampling
//! (10 to 25 steps) and Diffusion Posterior Sampling (DPS) inverse problem guidance.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for EDM2 / K-Diffusion samplers.
#[derive(Debug, Clone)]
pub struct EDMConfig {
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub rho: f32,
    pub num_steps: usize,
}

impl Default for EDMConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            rho: 7.0,
            num_steps: 15,
        }
    }
}

impl EDMConfig {
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps: num_steps.max(2),
            ..Default::default()
        }
    }

    /// Generates polynomial noise sequence $\sigma_0 > \sigma_1 > \dots > \sigma_{N-1} > 0$.
    pub fn build_sigmas(&self) -> Vec<f32> {
        // Backward-compatible clamped schedule: num_steps below 2 is
        // raised to 2 (matching new) and invalid parameters fall back
        // to defaults, so this never returns NaN sigmas. Prefer
        // try_build_sigmas when validation errors are actionable.
        let sigma_min = if self.sigma_min.is_finite() && self.sigma_min > 0.0 {
            self.sigma_min
        } else {
            0.002
        };
        let sigma_max = if self.sigma_max.is_finite() && self.sigma_max > sigma_min {
            self.sigma_max
        } else {
            sigma_min + 1.0
        };
        let sane = EDMConfig {
            num_steps: self.num_steps.max(2),
            sigma_min,
            sigma_max,
            rho: if self.rho.is_finite() && self.rho > 0.0 {
                self.rho
            } else {
                7.0
            },
        };
        sane.try_build_sigmas()
            .unwrap_or_else(|_| vec![sigma_max, sigma_min, 0.0])
    }

    /// Validated polynomial noise sequence with N+1 entries ending at 0.
    /// Rejects degenerate step counts and non-finite schedule parameters
    /// instead of producing NaN sigmas or a zero-only schedule.
    pub fn try_build_sigmas(&self) -> Result<Vec<f32>> {
        let n = self.num_steps;
        if n < 2 {
            return Err(Error::Msg(format!(
                "EDMConfig needs num_steps >= 2, got {}",
                n
            )));
        }
        crate::limits::check_iterations("EDMConfig num_steps", n)?;
        if !self.rho.is_finite() || self.rho <= 0.0 {
            return Err(Error::Msg(format!(
                "EDM rho must be finite and > 0, got {}",
                self.rho
            )));
        }
        if !self.sigma_min.is_finite() || self.sigma_min <= 0.0 {
            return Err(Error::Msg(format!(
                "EDM sigma_min must be finite and > 0, got {}",
                self.sigma_min
            )));
        }
        if !self.sigma_max.is_finite() || self.sigma_max <= self.sigma_min {
            return Err(Error::Msg(format!(
                "EDM sigma_max ({}) must exceed sigma_min ({})",
                self.sigma_max, self.sigma_min
            )));
        }
        let inv_rho = 1.0 / self.rho;
        let min_inv = self.sigma_min.powf(inv_rho);
        let max_inv = self.sigma_max.powf(inv_rho);

        let mut sigmas = Vec::with_capacity(n + 1);
        for i in 0..n {
            let frac = (i as f32) / ((n - 1) as f32);
            let s = (max_inv + frac * (min_inv - max_inv)).powf(self.rho);
            if !s.is_finite() {
                return Err(Error::Msg(format!(
                    "EDM schedule produced non-finite sigma at step {}",
                    i
                )));
            }
            sigmas.push(s);
        }
        sigmas.push(0.0); // final step reaches clean signal
        Ok(sigmas)
    }
}

/// Continuous-Time K-Diffusion & EDM2 Solver.
#[derive(Debug, Clone)]
pub struct KDiffusionSampler {
    pub config: EDMConfig,
}

impl KDiffusionSampler {
    pub fn new(config: EDMConfig) -> Self {
        Self { config }
    }

    /// Samples from initial noise using Heun's 2nd-order predictor-corrector ODE solver.
    ///
    /// `denoise_fn`: closure evaluating denoiser $D(x, \sigma) \approx x_0$.
    pub fn sample_heun<F>(&self, initial_noise: &Tensor, mut denoise_fn: F) -> Result<Tensor>
    where
        F: FnMut(&Tensor, f32) -> Result<Tensor>,
    {
        let sigmas = self.config.try_build_sigmas()?;
        let num_steps = self.config.num_steps;

        // Scale initial noise by sigma_0
        let mut x = initial_noise.mul_scalar(sigmas[0] as f64)?;

        for i in 0..num_steps {
            let sigma_cur = sigmas[i];
            let sigma_next = sigmas[i + 1];

            // 1. Predictor step (Euler)
            let denoised = denoise_fn(&x, sigma_cur)?;
            let d_cur = x.sub(&denoised)?.div_scalar(sigma_cur as f64)?;

            let dt = (sigma_next - sigma_cur) as f64;
            let x_prime = x.add(&d_cur.mul_scalar(dt)?)?;

            if sigma_next <= 1e-6 {
                // Reached final noise level
                x = x_prime;
                break;
            }

            // 2. Corrector step (Heun 2nd-order average slope)
            let denoised_prime = denoise_fn(&x_prime, sigma_next)?;
            let d_prime = x_prime
                .sub(&denoised_prime)?
                .div_scalar(sigma_next as f64)?;
            let d_avg = d_cur.add(&d_prime)?.mul_scalar(0.5)?;

            x = x.add(&d_avg.mul_scalar(dt)?)?;
        }

        Ok(x)
    }

    /// Diffusion Posterior Sampling (DPS) for inverse problems (deblurring, super-resolution).
    ///
    /// Solves $y = A(x_0) + \epsilon$ via measurement gradient guidance.
    pub fn dps_step(
        &self,
        x_t: &Tensor,
        denoised_x0: &Tensor,
        measurement_error_grad: &Tensor,
        zeta: f32,
    ) -> Result<Tensor> {
        let guided_x0 = denoised_x0.sub(&measurement_error_grad.mul_scalar(zeta as f64)?)?;
        // Re-inject noise according to current step
        guided_x0.add(&x_t.sub(denoised_x0)?)
    }
}
