//! Susskind-Brown-Roberts-Susskind-Swingle-Zhao Holographic Complexity = Volume (CV) & Complexity = Action (CA) Duality (Susskind, 2016; Brown et al., 2016).
//!
//! ### 1. Late-Time Wheeler-DeWitt Action Growth Rate $d\mathcal{C}_A / dt = 2 M / (\pi \hbar)$ (Lloyd's Bound)
//! Under the Complexity = Action (CA) conjecture, the quantum circuit complexity $\mathcal{C}_A = I_{\mathrm{WDW}} / (\pi \hbar)$ of an eternal neutral AdS black hole of mass $M$ grows linearly at late times at the universal Lloyd rate:
//! $$\lim_{t \to \infty} \frac{d \mathcal{C}_A}{d t} = \frac{2 M}{\pi \hbar}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Susskind-Brown Holographic Complexity = Action / Volume Engine.
#[derive(Debug, Clone, Copy)]
pub struct HolographicComplexityActionVolume {
    /// Black hole mass-energy $M > 0$ (in natural units $\hbar = 1$).
    pub black_hole_mass_m: f64,
}

impl Default for HolographicComplexityActionVolume {
    fn default() -> Self {
        Self {
            black_hole_mass_m: PI,
        }
    }
}

impl HolographicComplexityActionVolume {
    /// Creates a new Holographic Complexity = Action/Volume engine.
    pub fn new(black_hole_mass_m: f64) -> Result<Self> {
        if black_hole_mass_m <= 0.0 || !black_hole_mass_m.is_finite() {
            return Err(Error::Msg("black_hole_mass_m must be > 0".to_string()));
        }
        Ok(Self { black_hole_mass_m })
    }

    /// Evaluates the asymptotic Lloyd complexity growth rate $2 M / \pi$ and accumulated action complexity $\Delta \mathcal{C}_A(t) = (2 M / \pi) t$.
    pub fn lloyd_growth_rate_and_complexity(&self, time_t: f64) -> Result<(f64, f64)> {
        if time_t < 0.0 || !time_t.is_finite() {
            return Err(Error::Msg("time_t must be >= 0".to_string()));
        }
        let rate = 2.0 * self.black_hole_mass_m / PI;
        Ok((rate, rate * time_t))
    }
}
