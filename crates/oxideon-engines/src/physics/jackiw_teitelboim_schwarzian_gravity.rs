//! Jackiw-Teitelboim (JT) 2D Dilaton Gravity, Boundary Schwarzian Action &Saad-Shenker-Stanford Spectral Density $\rho_0(E) \propto \sinh(2\pi\sqrt{2 C E})$ (Jackiw, 1985; Teitelboim, 1983; Saad, Shenker & Stanford, 2019).
//!
//! ### 1. Exact Disk Spectral Density $\rho_{\mathrm{JT}}(E) = \frac{C}{2\pi^2} e^{S_0} \sinh\!\left(2\pi \sqrt{2 C E}\right)$
//! Integrating out the bulk metric in $\mathrm{AdS}_2$ JT gravity reduces to the boundary Schwarzian theory with disk partition function and exact density of states $\rho_{\mathrm{JT}}(E)$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Jackiw-Teitelboim (JT) 2D Gravity & Schwarzian Spectral Engine.
#[derive(Debug, Clone, Copy)]
pub struct JackiwTeitelboimSchwarzianGravity {
    /// Schwarzian coupling scale $C > 0$.
    pub schwarzian_coupling_c: f64,
    /// Extremal topological entropy $S_0 \ge 0$.
    pub extremal_entropy_s0: f64,
}

impl Default for JackiwTeitelboimSchwarzianGravity {
    fn default() -> Self {
        Self {
            schwarzian_coupling_c: 1.0,
            extremal_entropy_s0: 2.0,
        }
    }
}

impl JackiwTeitelboimSchwarzianGravity {
    /// Creates a new JT Schwarzian Gravity engine.
    pub fn new(schwarzian_coupling_c: f64, extremal_entropy_s0: f64) -> Result<Self> {
        if schwarzian_coupling_c <= 0.0 || extremal_entropy_s0 < 0.0 {
            return Err(Error::Msg(
                "schwarzian_coupling_c > 0 and extremal_entropy_s0 >= 0 required".to_string(),
            ));
        }
        Ok(Self {
            schwarzian_coupling_c,
            extremal_entropy_s0,
        })
    }

    /// Evaluates the exact disk density of states $\rho_{\mathrm{JT}}(E)$ and semiclassical Bekenstein-Hawking entropy $S(E) = S_0 + 2\pi\sqrt{2 C E}$.
    pub fn disk_spectral_density_and_entropy(&self, energy_e: f64) -> Result<(f64, f64)> {
        if energy_e <= 0.0 || !energy_e.is_finite() {
            return Err(Error::Msg("energy_e must be > 0".to_string()));
        }
        let arg = 2.0 * PI * (2.0 * self.schwarzian_coupling_c * energy_e).sqrt();
        let rho = (self.schwarzian_coupling_c / (2.0 * PI * PI))
            * self.extremal_entropy_s0.exp()
            * arg.sinh();
        let s_bh = self.extremal_entropy_s0 + arg;
        Ok((rho, s_bh))
    }
}
