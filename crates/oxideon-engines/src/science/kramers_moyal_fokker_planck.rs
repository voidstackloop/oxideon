//! Kramers-Moyal Expansion, Fokker-Planck PDE & Exact Ornstein-Uhlenbeck Green's Function Propagator (Risken, 1989).
//!
//! ### 1. Exact Propagator $P(x, t \mid x_0, 0)$ of the Ornstein-Uhlenbeck Fokker-Planck Equation
//! The Fokker-Planck PDE $\partial_t P = \gamma \partial_x (x P) + D \partial_x^2 P$ has the exact Gaussian Green's function
//! with mean $\mu(t) = x_0 e^{-\gamma t}$ and variance $\sigma^2(t) = \frac{D}{\gamma}(1 - e^{-2 \gamma t})$:
//! $$P(x, t \mid x_0, 0) = \frac{1}{\sqrt{2 \pi \sigma^2(t)}} \exp\left(-\frac{(x - x_0 e^{-\gamma t})^2}{2 \sigma^2(t)}\right)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Kramers-Moyal Fokker-Planck & Ornstein-Uhlenbeck Propagator Engine.
#[derive(Debug, Clone, Copy)]
pub struct FokkerPlanckOrnsteinUhlenbeck {
    /// Drift relaxation rate $\gamma > 0$.
    pub gamma: f64,
    /// Diffusion constant $D > 0$.
    pub diffusion_d: f64,
}

impl FokkerPlanckOrnsteinUhlenbeck {
    /// Creates a new Fokker-Planck Ornstein-Uhlenbeck engine.
    pub fn new(gamma: f64, diffusion_d: f64) -> Result<Self> {
        if gamma <= 0.0 || diffusion_d <= 0.0 {
            return Err(Error::Msg("gamma and diffusion_d must be > 0".to_string()));
        }
        Ok(Self { gamma, diffusion_d })
    }

    /// Evaluates the exact Fokker-Planck transition probability density $P(x, t \mid x_0, 0)$ for $t > 0$.
    pub fn propagator_density(&self, x: f64, x0: f64, t: f64) -> Result<f64> {
        if t <= 0.0 {
            return Err(Error::Msg("Time t must be > 0".to_string()));
        }
        let mu = x0 * (-self.gamma * t).exp();
        let var = (self.diffusion_d / self.gamma) * (1.0 - (-2.0 * self.gamma * t).exp());
        let dx = x - mu;
        Ok((-0.5 * dx * dx / var).exp() / (2.0 * PI * var).sqrt())
    }
}
