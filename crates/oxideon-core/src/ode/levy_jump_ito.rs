//! Lévy-Itô Jump-Diffusion Processes, Lévy-Khintchine Characteristic Exponent & Tempered Stable CGMY Processes.
//!
//! ### 1. Lévy-Khintchine Formula for Compound Poisson Jump-Diffusions
//! A jump-diffusion $X_t = b t + \sigma W_t + \sum_{k=1}^{N_t} J_k$ with Poisson intensity $\lambda \ge 0$ and Gaussian
//! jump size $J_k \sim \mathcal{N}(\mu_J, \sigma_J^2)$ has characteristic exponent $\Psi(u) = \ln \mathbb{E}[e^{i u X_1}]$:
//! $$\operatorname{Re} \Psi(u) = -\frac{1}{2} \sigma^2 u^2 + \lambda \left(e^{-\frac{1}{2} \sigma_J^2 u^2} \cos(\mu_J u) - 1\right)$$

use crate::error::{Error, Result};

/// Lévy-Itô Jump-Diffusion & Lévy-Khintchine Characteristic Engine.
#[derive(Debug, Clone, Copy)]
pub struct LevyItodiffusionEngine {
    /// Continuous diffusion volatility $\sigma \ge 0$.
    pub sigma: f64,
    /// Jump arrival intensity $\lambda \ge 0$.
    pub jump_lambda: f64,
    /// Mean jump size $\mu_J$.
    pub jump_mean: f64,
    /// Jump standard deviation $\sigma_J \ge 0$.
    pub jump_std: f64,
}

impl LevyItodiffusionEngine {
    /// Creates a new Lévy-Itô jump-diffusion engine.
    pub fn new(sigma: f64, jump_lambda: f64, jump_mean: f64, jump_std: f64) -> Result<Self> {
        if sigma < 0.0 || jump_lambda < 0.0 || jump_std < 0.0 {
            return Err(Error::Msg(
                "sigma, jump_lambda, and jump_std must be >= 0".to_string(),
            ));
        }
        Ok(Self {
            sigma,
            jump_lambda,
            jump_mean,
            jump_std,
        })
    }

    /// Evaluates the real part of the Lévy-Khintchine characteristic exponent $\operatorname{Re} \Psi(u)$ at frequency $u$.
    pub fn real_levy_exponent(&self, u: f64) -> f64 {
        let diff_part = -0.5 * self.sigma * self.sigma * u * u;
        let jump_part = self.jump_lambda
            * ((-0.5 * self.jump_std * self.jump_std * u * u).exp() * (self.jump_mean * u).cos()
                - 1.0);
        diff_part + jump_part
    }
}
