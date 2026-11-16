//! Arıkan Channel Polarization & Capacity-Achieving Polar Codes (Arıkan, 2009).
//!
//! ### 1. Recursive Bhattacharyya Parameter Evolution $(Z^-, Z^+) = (2Z - Z^2, Z^2)$ on BEC
//! Under one step of Arıkan's $G_2 = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$ transform on a Binary Erasure Channel $\mathrm{BEC}(\epsilon)$:
//! $$Z(W^-) = 2 Z(W) - Z(W)^2, \qquad Z(W^+) = Z(W)^2, \qquad Z(W^-) + Z(W^+) = 2 Z(W)$$
//! preserving average mutual information $I(W^-) + I(W^+) = 2(1 - Z(W))$ while driving synthetic channels toward $0$ or $1$.

use crate::error::{Error, Result};

/// Arıkan Channel Polarization & Polar Code Bhattacharyya Engine.
#[derive(Debug, Clone, Copy)]
pub struct ArikanPolarChannelPolarization {
    /// Initial erasure / Bhattacharyya parameter $Z_0 \in (0, 1)$.
    pub initial_z0: f64,
}

impl Default for ArikanPolarChannelPolarization {
    fn default() -> Self {
        Self { initial_z0: 0.5 }
    }
}

impl ArikanPolarChannelPolarization {
    /// Creates a new Arıkan Channel Polarization engine.
    pub fn new(initial_z0: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&initial_z0) {
            return Err(Error::Msg("initial_z0 must be in [0, 1]".to_string()));
        }
        Ok(Self { initial_z0 })
    }

    /// Evaluates the degraded/upgraded Bhattacharyya pair $(Z^-, Z^+) = (2Z - Z^2, Z^2)$ and polarization gap $Z^- - Z^+$.
    pub fn butterfly_bhattacharyya_split(&self) -> (f64, f64, f64) {
        let z = self.initial_z0;
        let z_minus = 2.0 * z - z * z;
        let z_plus = z * z;
        (z_minus, z_plus, z_minus - z_plus)
    }
}
