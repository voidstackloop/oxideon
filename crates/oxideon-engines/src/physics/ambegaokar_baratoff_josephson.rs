//! Ambegaokar-Baratoff Tunnel Junction Critical Current $I_c R_n$ & AC Josephson Shapiro Steps (Josephson, 1962; Ambegaokar & Baratoff, 1963).
//!
//! ### 1. Ambegaokar-Baratoff $I_c R_n$ Product
//! For a symmetric superconductor-insulator-superconductor (SIS) Josephson junction with BCS gap $\Delta(T)$ and normal resistance $R_n$:
//! $$I_c(T) R_n = \frac{\pi \Delta(T)}{2 e} \tanh\!\left(\frac{\Delta(T)}{2 k_B T}\right)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Ambegaokar-Baratoff Josephson Critical Current & Shapiro Step Engine.
#[derive(Debug, Clone, Copy)]
pub struct AmbegaokarBaratoffJosephson {
    /// Superconducting BCS energy gap $\Delta > 0$ (in natural units $e = k_B = 1$).
    pub gap_delta: f64,
}

impl Default for AmbegaokarBaratoffJosephson {
    fn default() -> Self {
        Self { gap_delta: 1.0 }
    }
}

impl AmbegaokarBaratoffJosephson {
    /// Creates a new Ambegaokar-Baratoff Josephson engine.
    pub fn new(gap_delta: f64) -> Result<Self> {
        if gap_delta <= 0.0 || !gap_delta.is_finite() {
            return Err(Error::Msg("gap_delta must be > 0".to_string()));
        }
        Ok(Self { gap_delta })
    }

    /// Evaluates the $I_c R_n$ product at temperature $T > 0$ and the zero-temperature limit $\pi \Delta / 2$.
    pub fn ic_rn_product(&self, temperature_t: f64) -> Result<(f64, f64)> {
        if temperature_t <= 0.0 {
            return Err(Error::Msg("temperature_t must be > 0".to_string()));
        }
        let zero_t = 0.5 * PI * self.gap_delta;
        let ic_rn = zero_t * (self.gap_delta / (2.0 * temperature_t)).tanh();
        Ok((ic_rn, zero_t))
    }
}
