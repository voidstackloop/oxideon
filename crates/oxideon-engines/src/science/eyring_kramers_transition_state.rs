//! Eyring-Polanyi Transition State Theory (TST) & Wigner Parabolic Quantum Tunneling Correction (Eyring, 1935; Wigner, 1932).
//!
//! ### 1. Eyring Rate Constant & Wigner Semiclassical Tunneling Transmission $\kappa_{\mathrm{Wigner}}(T)$
//! Across a saddle point with imaginary barrier frequency $i \omega^\ddagger$ ($\omega^\ddagger > 0$) and activation free energy $\Delta G^\ddagger$,
//! Wigner's leading semiclassical quantum tunneling transmission factor is:
//! $$\kappa_{\mathrm{Wigner}}(T) = 1 + \frac{1}{24}\left(\frac{\hbar \omega^\ddagger}{k_B T}\right)^2$$

use crate::error::{Error, Result};

/// Eyring Transition State Theory & Wigner Quantum Tunneling Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct EyringTransitionStateEngine;

impl EyringTransitionStateEngine {
    /// Creates a new Eyring Transition State engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates Wigner's quantum tunneling transmission coefficient $\kappa_{\mathrm{Wigner}} = 1 + \frac{1}{24} u_b^2$ where $u_b = \hbar \omega^\ddagger / (k_B T) \ge 0$.
    pub fn wigner_tunneling_factor(reduced_barrier_frequency_ub: f64) -> Result<f64> {
        if reduced_barrier_frequency_ub < 0.0 {
            return Err(Error::Msg(
                "reduced_barrier_frequency_ub must be >= 0".to_string(),
            ));
        }
        Ok(1.0 + (reduced_barrier_frequency_ub * reduced_barrier_frequency_ub) / 24.0)
    }
}
