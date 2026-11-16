//! Sweet-Parker & Petschek Magnetic Reconnection Rate & Plasmoid Instability Threshold (Sweet, 1958; Parker, 1957; Petschek, 1964; Loureiro et al., 2007).
//!
//! ### 1. Sweet-Parker vs. Petschek Dimensionless Reconnection Rate $M_A = v_{\mathrm{in}} / v_A$
//! For Lundquist number $S = L v_A / \eta$, the Sweet-Parker current sheet has reconnection rate $M_{\mathrm{SP}} = S^{-1/2}$,
//! Petschek slow-shock reconnection achieves $M_{\mathrm{Pet}} = \frac{\pi}{8 \ln S}$, and plasmoid tearing instability sets in above $S_c \approx 10^4$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Sweet-Parker, Petschek & Plasmoid Magnetic Reconnection Engine.
#[derive(Debug, Clone, Copy)]
pub struct SweetParkerPetschekReconnection {
    /// Lundquist number $S = L v_A / \eta > 1$.
    pub lundquist_s: f64,
}

impl Default for SweetParkerPetschekReconnection {
    fn default() -> Self {
        Self { lundquist_s: 1.0e6 }
    }
}

impl SweetParkerPetschekReconnection {
    /// Creates a new Sweet-Parker-Petschek Magnetic Reconnection engine.
    pub fn new(lundquist_s: f64) -> Result<Self> {
        if lundquist_s <= 1.0 || !lundquist_s.is_finite() {
            return Err(Error::Msg("Lundquist number S must be > 1".to_string()));
        }
        Ok(Self { lundquist_s })
    }

    /// Evaluates the Sweet-Parker rate $S^{-1/2}$, Petschek fast rate $\pi / (8 \ln S)$, and plasmoid-unstable regime check ($S > 10^4$).
    pub fn reconnection_rates_and_plasmoid_regime(&self) -> (f64, f64, bool) {
        let sp_rate = 1.0 / self.lundquist_s.sqrt();
        let pet_rate = PI / (8.0 * self.lundquist_s.ln());
        (sp_rate, pet_rate, self.lundquist_s > 1.0e4)
    }
}
