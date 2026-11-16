//! Tsu-Esaki Double-Barrier Resonant Tunneling Diode (RTD) & Breit-Wigner Transmission Resonance (Tsu & Esaki, 1973).
//!
//! ### 1. Breit-Wigner Quantum Well Resonance Transmission
//! For a double-barrier quantum well with quasi-bound resonance energy $E_r$ and left/right barrier escape widths $\Gamma_L, \Gamma_R > 0$ ($\Gamma = \Gamma_L + \Gamma_R$):
//! $$T(E) = \frac{\Gamma_L \Gamma_R}{(E - E_r)^2 + (\Gamma_L + \Gamma_R)^2 / 4}$$
//! Symmetric barriers ($\Gamma_L = \Gamma_R$) achieve unit transmission $T(E_r) = 1$ at resonance
//! (the factor 4 appears only in the peak form $T_{\max} = 4\Gamma_L\Gamma_R/(\Gamma_L+\Gamma_R)^2$).

use crate::error::{Error, Result};

/// Tsu-Esaki Resonant Tunneling & Breit-Wigner Double-Barrier Engine.
#[derive(Debug, Clone, Copy)]
pub struct TsuEsakiResonantTunneling {
    /// Quasi-bound resonance energy $E_r$.
    pub resonance_er: f64,
    /// Left barrier tunneling width $\Gamma_L > 0$.
    pub gamma_l: f64,
    /// Right barrier tunneling width $\Gamma_R > 0$.
    pub gamma_r: f64,
}

impl Default for TsuEsakiResonantTunneling {
    fn default() -> Self {
        Self {
            resonance_er: 0.2,
            gamma_l: 0.01,
            gamma_r: 0.01,
        }
    }
}

impl TsuEsakiResonantTunneling {
    /// Creates a new Tsu-Esaki Resonant Tunneling engine.
    pub fn new(resonance_er: f64, gamma_l: f64, gamma_r: f64) -> Result<Self> {
        if gamma_l <= 0.0 || gamma_r <= 0.0 {
            return Err(Error::Msg("gamma_l and gamma_r must be > 0".to_string()));
        }
        Ok(Self {
            resonance_er,
            gamma_l,
            gamma_r,
        })
    }

    /// Evaluates Breit-Wigner transmission $T(E)$ at electron energy $E$ and the peak resonance transmission $T_{\mathrm{max}} = 4 \Gamma_L \Gamma_R / (\Gamma_L + \Gamma_R)^2$.
    pub fn transmission_and_peak(&self, energy_e: f64) -> (f64, f64) {
        let gamma_tot = self.gamma_l + self.gamma_r;
        let num = self.gamma_l * self.gamma_r;
        let t_e = num / ((energy_e - self.resonance_er).powi(2) + 0.25 * gamma_tot * gamma_tot);
        let t_max = 4.0 * num / (gamma_tot * gamma_tot);
        (t_e, t_max)
    }
}
