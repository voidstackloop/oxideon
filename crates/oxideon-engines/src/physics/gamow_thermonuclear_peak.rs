//! Gamow Thermonuclear Quantum Tunneling Peak & Stellar Reaction Window $\Delta E_0$ (Gamow, 1928; Bethe, 1939).
//!
//! ### 1. Gamow Peak Energy $E_0$ & Effective Tunneling Window Width $\Delta$
//! Convolution of the Maxwell-Boltzmann tail $\exp(-E / k_B T)$ with the Coulomb WKB barrier penetration $\exp(-\sqrt{E_G / E})$
//! peaks at the Gamow saddle point:
//! $$E_0 = \left(\frac{E_G (k_B T)^2}{4}\right)^{1/3}, \qquad \Delta = \frac{4}{\sqrt{3}} \sqrt{E_0 k_B T}, \qquad \tau = \frac{3 E_0}{k_B T}$$

use crate::error::{Error, Result};

/// Gamow Thermonuclear Tunneling Peak & Bethe CNO/pp Window Engine.
#[derive(Debug, Clone, Copy)]
pub struct GamowThermonuclearPeakEngine {
    /// Gamow barrier energy $E_G = 2 \mu c^2 (\pi \alpha Z_1 Z_2)^2 > 0$ (keV).
    pub gamow_energy_eg_kev: f64,
}

impl Default for GamowThermonuclearPeakEngine {
    fn default() -> Self {
        Self {
            gamow_energy_eg_kev: 493.0,
        }
    }
}

impl GamowThermonuclearPeakEngine {
    /// Creates a new Gamow Thermonuclear Peak engine.
    pub fn new(gamow_energy_eg_kev: f64) -> Result<Self> {
        if gamow_energy_eg_kev <= 0.0 || !gamow_energy_eg_kev.is_finite() {
            return Err(Error::Msg("gamow_energy_eg_kev must be > 0".to_string()));
        }
        Ok(Self {
            gamow_energy_eg_kev,
        })
    }

    /// Evaluates the Gamow peak energy $E_0$, Gaussian window width $\Delta$, and Sommerfeld-Gamow exponent $\tau = 3 E_0 / (k_B T)$ at thermal energy $k_B T > 0$ (keV).
    pub fn gamow_peak_width_and_tau(&self, k_b_t_kev: f64) -> Result<(f64, f64, f64)> {
        if k_b_t_kev <= 0.0 {
            return Err(Error::Msg("k_b_t_kev must be > 0".to_string()));
        }
        let e0 = (0.25 * self.gamow_energy_eg_kev * k_b_t_kev * k_b_t_kev).cbrt();
        let width_delta = (4.0 / 3.0_f64.sqrt()) * (e0 * k_b_t_kev).sqrt();
        let tau = 3.0 * e0 / k_b_t_kev;
        Ok((e0, width_delta, tau))
    }
}
