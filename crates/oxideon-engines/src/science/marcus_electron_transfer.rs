//! Rudolph Marcus Theory of Electron Transfer, Reorganization Energy $\lambda$ & Marcus Inverted Region (Marcus, 1956 — 1992 Nobel Prize).
//!
//! ### 1. Parabolic Diabatic Free Energy Surfaces & Marcus Rate Constant $k_{\mathrm{ET}}$
//! For an outer-sphere electron transfer with solvent reorganization energy $\lambda > 0$ and driving force $\Delta G^0$,
//! the activation barrier is:
//! $$\Delta G^\ddagger = \frac{(\lambda + \Delta G^0)^2}{4 \lambda}$$
//! When $-\Delta G^0 = \lambda$, the reaction is activationless ($\Delta G^\ddagger = 0$); for $-\Delta G^0 > \lambda$
//! (the **Marcus Inverted Region**), increasing the thermodynamic driving force *decreases* the reaction rate!

use crate::error::{Error, Result};

/// Rudolph Marcus Electron Transfer & Inverted Region Engine.
#[derive(Debug, Clone, Copy)]
pub struct MarcusElectronTransferEngine {
    /// Reorganization energy $\lambda > 0$ (in $\mathrm{eV}$).
    pub reorganization_lambda: f64,
}

impl MarcusElectronTransferEngine {
    /// Creates a new Marcus Electron Transfer engine.
    pub fn new(reorganization_lambda: f64) -> Result<Self> {
        if reorganization_lambda <= 0.0 {
            return Err(Error::Msg("reorganization_lambda must be > 0".to_string()));
        }
        Ok(Self {
            reorganization_lambda,
        })
    }

    /// Computes the Marcus activation free energy barrier $\Delta G^\ddagger = \frac{(\lambda + \Delta G^0)^2}{4 \lambda}$ and checks if in the inverted region ($-\Delta G^0 > \lambda$).
    pub fn activation_barrier_and_is_inverted(&self, delta_g0: f64) -> (f64, bool) {
        let sum = self.reorganization_lambda + delta_g0;
        let barrier = (sum * sum) / (4.0 * self.reorganization_lambda);
        (barrier, -delta_g0 > self.reorganization_lambda)
    }
}
