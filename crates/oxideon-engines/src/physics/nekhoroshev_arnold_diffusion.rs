//! Nekhoroshev Exponential Stability Theorem & Arnold Diffusion Along Resonance Webs (Arnold, 1964; Nekhoroshev, 1977).
//!
//! ### 1. Lochak-Nekhoroshev Optimal Steepness Exponents $a = b = \frac{1}{2n}$
//! For an $n$-degree-of-freedom quasi-convex nearly integrable Hamiltonian $H(I, \theta) = h(I) + \epsilon f(I, \theta)$ ($0 < \epsilon < \epsilon_0$),
//! Nekhoroshev's Theorem (with Lochak's periodic-orbit exponents $a = b = \frac{1}{2n}$) guarantees that for ALL initial actions $I(0)$:
//! $$\|I(t) - I(0)\| \le I_0 \left(\frac{\epsilon}{\epsilon_0}\right)^{\frac{1}{2 n}} \quad \text{for all} \quad |t| \le T_0 \exp\left[\left(\frac{\epsilon_0}{\epsilon}\right)^{\frac{1}{2 n}}\right]$$

use crate::error::{Error, Result};

/// Nekhoroshev Exponential Action Stability & Arnold Diffusion Engine.
#[derive(Debug, Clone, Copy)]
pub struct NekhoroshevExponentialStability {
    /// Degrees of freedom $n \ge 2$ (Arnold diffusion occurs for $n \ge 3$, or $n = 2.5$ time-periodic).
    pub dof_n: usize,
    /// Critical perturbation threshold $\epsilon_0 > 0$.
    pub eps_0: f64,
}

impl NekhoroshevExponentialStability {
    /// Creates a new Nekhoroshev Exponential Stability engine.
    pub fn new(dof_n: usize, eps_0: f64) -> Result<Self> {
        if dof_n < 2 || eps_0 <= 0.0 {
            return Err(Error::Msg("dof_n >= 2 and eps_0 > 0 required".to_string()));
        }
        Ok(Self { dof_n, eps_0 })
    }

    /// Evaluates the Lochak-Nekhoroshev stability exponent $a = 1 / (2n)$, action drift radius $(\epsilon / \epsilon_0)^a$, and exponential confinement time factor $\exp((\epsilon_0 / \epsilon)^a)$.
    pub fn nekhoroshev_drift_and_time_factor(&self, perturbation_eps: f64) -> Result<(f64, f64)> {
        if perturbation_eps <= 0.0 || perturbation_eps >= self.eps_0 {
            return Err(Error::Msg(
                "perturbation_eps must be in (0, eps_0)".to_string(),
            ));
        }
        let a = 1.0 / (2.0 * (self.dof_n as f64));
        let ratio = (self.eps_0 / perturbation_eps).powf(a);
        Ok((1.0 / ratio, ratio.exp()))
    }
}
