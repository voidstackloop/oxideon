//! Feshbach Magneto-Association Resonance, Tunable $s$-Wave Scattering Length & Unitary Fermi Gas Bertsch Parameter (Feshbach, 1958; Regal, Greiner & Jin, 2004).
//!
//! ### 1. Magnetic Feshbach Resonance Formula
//! Near a magnetic Feshbach resonance at field $B_0$ with width $\Delta B$ and background scattering length $a_{\mathrm{bg}}$:
//! $$a(B) = a_{\mathrm{bg}} \left(1 - \frac{\Delta B}{B - B_0}\right)$$

use crate::error::{Error, Result};

/// Feshbach Resonance & Unitary BEC-BCS Crossover Engine.
#[derive(Debug, Clone, Copy)]
pub struct FeshbachScatteringResonance {
    /// Background scattering length $a_{\mathrm{bg}}$ (in Bohr radii $a_0$).
    pub a_bg: f64,
    /// Resonance center field $B_0$ (Gauss).
    pub b0_gauss: f64,
    /// Resonance width $\Delta B$ (Gauss).
    pub delta_b_gauss: f64,
}

impl Default for FeshbachScatteringResonance {
    fn default() -> Self {
        Self {
            a_bg: 100.0,
            b0_gauss: 832.0,
            delta_b_gauss: -300.0,
        }
    }
}

impl FeshbachScatteringResonance {
    /// Creates a new Feshbach Scattering Resonance engine.
    pub fn new(a_bg: f64, b0_gauss: f64, delta_b_gauss: f64) -> Result<Self> {
        if a_bg == 0.0 || delta_b_gauss == 0.0 {
            return Err(Error::Msg(
                "a_bg and delta_b_gauss must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            a_bg,
            b0_gauss,
            delta_b_gauss,
        })
    }

    /// Evaluates the $s$-wave scattering length $a(B)$ and checks if the system is on the BEC side ($a(B) > 0$) vs. BCS side ($a(B) < 0$).
    pub fn scattering_length_and_bec_side(&self, field_b_gauss: f64) -> Result<(f64, bool)> {
        let diff = field_b_gauss - self.b0_gauss;
        if diff.abs() < 1e-12 {
            return Err(Error::Msg(
                "Exact unitarity pole B == B0 has divergent scattering length".to_string(),
            ));
        }
        let a_b = self.a_bg * (1.0 - self.delta_b_gauss / diff);
        Ok((a_b, a_b > 0.0))
    }
}
