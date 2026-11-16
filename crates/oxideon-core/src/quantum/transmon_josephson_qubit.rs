//! Koch-Girvin Circuit QED Transmon Qubit, Josephson-to-Charging Ratio $E_J / E_C$ & Exponential Charge Dispersion Suppression (Koch et al., 2007).
//!
//! ### 1. Transmon Plasma Frequency $\omega_{01}$ & Anharmonicity $\alpha$
//! In the transmon regime $E_J / E_C \gg 1$, the qubit transition frequency and negative anharmonicity are:
//! $$\hbar \omega_{01} \approx \sqrt{8 E_J E_C} - E_C, \qquad \alpha = \omega_{12} - \omega_{01} \approx -E_C$$
//! while peak-to-peak charge dispersion decays exponentially as $\epsilon_0 \propto \exp(-\sqrt{8 E_J / E_C})$.

use crate::error::{Error, Result};

/// Koch-Girvin Circuit QED Transmon Superconducting Qubit Engine.
#[derive(Debug, Clone, Copy)]
pub struct TransmonCircuitQEDQubit {
    /// Josephson energy $E_J > 0$ (GHz).
    pub e_j_ghz: f64,
    /// Charging energy $E_C > 0$ (GHz).
    pub e_c_ghz: f64,
}

impl Default for TransmonCircuitQEDQubit {
    fn default() -> Self {
        Self {
            e_j_ghz: 15.0,
            e_c_ghz: 0.25,
        }
    }
}

impl TransmonCircuitQEDQubit {
    /// Creates a new Transmon Circuit QED Qubit engine.
    pub fn new(e_j_ghz: f64, e_c_ghz: f64) -> Result<Self> {
        if e_j_ghz <= 0.0 || e_c_ghz <= 0.0 || e_j_ghz <= e_c_ghz {
            return Err(Error::Msg(
                "0 < e_c_ghz < e_j_ghz required for transmon regime".to_string(),
            ));
        }
        Ok(Self { e_j_ghz, e_c_ghz })
    }

    /// Evaluates the qubit frequency $\omega_{01} = \sqrt{8 E_J E_C} - E_C$, anharmonicity $\alpha = -E_C$, and WKB charge dispersion suppression factor $\exp(-\sqrt{8 E_J / E_C})$.
    pub fn frequency_anharmonicity_and_dispersion(&self) -> (f64, f64, f64) {
        let omega_01 = (8.0 * self.e_j_ghz * self.e_c_ghz).sqrt() - self.e_c_ghz;
        let anharmonicity = -self.e_c_ghz;
        let wkb_suppression = (-(8.0 * self.e_j_ghz / self.e_c_ghz).sqrt()).exp();
        (omega_01, anharmonicity, wkb_suppression)
    }
}
