//! Zimm-Bragg Alpha-Helix–Coil Statistical Mechanics & Transfer Matrix Eigenvalues (Zimm & Bragg, 1959; Ramachandran, 1963).
//!
//! ### 1. Zimm-Bragg $2 \times 2$ Transfer Matrix & Fractional Helicity $\theta_H$
//! With helix propagation weight $s > 0$ and nucleation cooperativity parameter $\sigma \ll 1$, the Zimm-Bragg transfer matrix
//! $M = \begin{pmatrix} 1 & \sigma s \\ 1 & s \end{pmatrix}$ has largest eigenvalue and infinite-chain fractional helicity:
//! $$\lambda_+ = \frac{1 + s + \sqrt{(1 - s)^2 + 4 \sigma s}}{2}, \qquad \theta_H = \frac{1}{2} + \frac{s - 1}{2\sqrt{(1 - s)^2 + 4 \sigma s}}$$

use crate::error::{Error, Result};

/// Zimm-Bragg Helix-Coil Transition & Ramachandran Conformational Engine.
#[derive(Debug, Clone, Copy)]
pub struct ZimmBraggRamachandranHelix {
    /// Nucleation cooperativity parameter $\sigma \in (0, 1]$.
    pub cooperativity_sigma: f64,
}

impl Default for ZimmBraggRamachandranHelix {
    fn default() -> Self {
        Self {
            cooperativity_sigma: 1.0e-3,
        }
    }
}

impl ZimmBraggRamachandranHelix {
    /// Creates a new Zimm-Bragg Ramachandran Helix engine.
    pub fn new(cooperativity_sigma: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&cooperativity_sigma) || cooperativity_sigma == 0.0 {
            return Err(Error::Msg(
                "cooperativity_sigma must be in (0, 1]".to_string(),
            ));
        }
        Ok(Self {
            cooperativity_sigma,
        })
    }

    /// Evaluates the leading transfer-matrix eigenvalue $\lambda_+$ and equilibrium helix fraction $\theta_H \in (0, 1)$ at propagation parameter $s > 0$.
    pub fn eigenvalue_and_helix_fraction(&self, propagation_s: f64) -> Result<(f64, f64)> {
        if propagation_s <= 0.0 {
            return Err(Error::Msg("propagation_s must be > 0".to_string()));
        }
        let disc =
            ((1.0 - propagation_s).powi(2) + 4.0 * self.cooperativity_sigma * propagation_s).sqrt();
        let lambda_plus = 0.5 * (1.0 + propagation_s + disc);
        let theta_h = 0.5 + 0.5 * (propagation_s - 1.0) / disc;
        Ok((lambda_plus, theta_h))
    }
}
