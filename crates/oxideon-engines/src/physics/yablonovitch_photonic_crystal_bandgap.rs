//! Yablonovitch-John Photonic Crystal Bandgaps & Quarter-Wave Distributed Bragg Reflector Stopband (Yablonovitch, 1987; John, 1987).
//!
//! ### 1. Exact Quarter-Wave Bragg Stopband Width & $N$-Bilayer Reflectance
//! For a 1D periodic photonic crystal of alternating quarter-wave dielectric layers ($n_H d_H = n_L d_L = \lambda_0 / 4$),
//! the fractional photonic bandgap width $\Delta \omega / \omega_0$ and normal-incidence peak reflectance $R_N$ for $N$ bilayers are:
//! $$\frac{\Delta \omega}{\omega_0} = \frac{4}{\pi} \arcsin\left(\frac{|n_H - n_L|}{n_H + n_L}\right), \qquad R_N = \left(\frac{1 - (n_H / n_L)^{2 N}}{1 + (n_H / n_L)^{2 N}}\right)^2$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Yablonovitch Photonic Crystal Bandgap & Distributed Bragg Reflector Engine.
#[derive(Debug, Clone, Copy)]
pub struct PhotonicBraggQuarterWaveStack {
    /// High refractive index $n_H > n_L$.
    pub n_high: f64,
    /// Low refractive index $n_L > 0$.
    pub n_low: f64,
}

impl PhotonicBraggQuarterWaveStack {
    /// Creates a new Photonic Crystal Quarter-Wave Stack engine.
    pub fn new(n_high: f64, n_low: f64) -> Result<Self> {
        if n_low <= 0.0 || n_high <= n_low {
            return Err(Error::Msg("Requires 0 < n_low < n_high".to_string()));
        }
        Ok(Self { n_high, n_low })
    }

    /// Computes the fractional photonic stopband width $\Delta \omega / \omega_0$ and $N$-period peak reflectance $R_N \in (0, 1)$.
    pub fn fractional_bandgap_and_reflectance(&self, num_periods_n: u32) -> (f64, f64) {
        let contrast = (self.n_high - self.n_low) / (self.n_high + self.n_low);
        let gap_ratio = (4.0 / PI) * contrast.asin();
        let ratio_2n = (self.n_high / self.n_low).powi(2 * (num_periods_n as i32));
        let r_n = ((1.0 - ratio_2n) / (1.0 + ratio_2n)).powi(2);
        (gap_ratio, r_n)
    }
}
