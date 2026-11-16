//! Shockley-Queisser Detailed Balance Photovoltaic Limit & Ultimate Spectrum Efficiency (Shockley & Queisser, 1961).
//!
//! ### 1. Ultimate Spectrum Efficiency $\eta_u(x_g)$
//! For dimensionless bandgap $x_g = E_g / (k_B T_s)$, the ultimate absorption-recombination spectrum efficiency is:
//! $$\eta_u(x_g) = \frac{x_g \int_{x_g}^\infty \frac{x^2}{e^x - 1} dx}{\int_0^\infty \frac{x^3}{e^x - 1} dx} = \frac{15}{\pi^4} x_g \int_{x_g}^\infty \frac{x^2}{e^x - 1} dx$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Shockley-Queisser Detailed-Balance Photovoltaic Efficiency Engine.
#[derive(Debug, Clone, Copy)]
pub struct ShockleyQueisserPhotovoltaic {
    /// Solar blackbody temperature $k_B T_s > 0$ (eV, typically $0.5$ eV for $5778$ K).
    pub k_b_t_sun_ev: f64,
}

impl Default for ShockleyQueisserPhotovoltaic {
    fn default() -> Self {
        Self { k_b_t_sun_ev: 0.5 }
    }
}

impl ShockleyQueisserPhotovoltaic {
    /// Creates a new Shockley-Queisser Photovoltaic engine.
    pub fn new(k_b_t_sun_ev: f64) -> Result<Self> {
        if k_b_t_sun_ev <= 0.0 || !k_b_t_sun_ev.is_finite() {
            return Err(Error::Msg("k_b_t_sun_ev must be > 0".to_string()));
        }
        Ok(Self { k_b_t_sun_ev })
    }

    /// Evaluates the Shockley-Queisser ultimate efficiency $\eta_u(E_g)$ at bandgap $E_g > 0$ (peaks near $x_g \approx 2.17$ at $\approx 43.9\%$).
    pub fn ultimate_spectrum_efficiency(&self, bandgap_eg_ev: f64) -> Result<f64> {
        if bandgap_eg_ev <= 0.0 {
            return Err(Error::Msg("bandgap_eg_ev must be > 0".to_string()));
        }
        let xg = bandgap_eg_ev / self.k_b_t_sun_ev;
        let x_max = xg + 30.0;
        let steps = 400usize;
        let dx = (x_max - xg) / (steps as f64);
        let mut integral = 0.0;
        for i in 0..steps {
            let x = xg + ((i as f64) + 0.5) * dx;
            integral += x * x / (x.exp() - 1.0) * dx;
        }
        Ok((15.0 / PI.powi(4)) * xg * integral)
    }
}
