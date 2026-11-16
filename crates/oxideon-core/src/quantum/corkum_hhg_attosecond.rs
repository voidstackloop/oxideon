//! Corkum Three-Step Recollision Model, High-Harmonic Generation (HHG) Cutoff & Keldysh Parameter (Keldysh, 1965; Corkum, 1993).
//!
//! ### 1. Ponderomotive Energy $U_p$, HHG Cutoff $I_p + 3.17 U_p$ & Keldysh Parameter $\gamma_K$
//! In atomic units ($e = m_e = \hbar = 1$), a laser field of amplitude $E_0$ and frequency $\omega_0$ acting on an atom with ionization potential $I_p$ yields:
//! $$U_p = \frac{E_0^2}{4 \omega_0^2}, \qquad E_{\mathrm{cutoff}} = I_p + 3.17 U_p, \qquad \gamma_K = \sqrt{\frac{I_p}{2 U_p}}$$

use crate::error::{Error, Result};

/// Corkum Three-Step High-Harmonic Generation & Attosecond Cutoff Engine.
#[derive(Debug, Clone, Copy)]
pub struct CorkumHighHarmonicAttosecond {
    /// Atomic ionization potential $I_p > 0$ (eV or atomic units).
    pub ionization_potential_ip: f64,
}

impl Default for CorkumHighHarmonicAttosecond {
    fn default() -> Self {
        Self {
            ionization_potential_ip: 15.76,
        }
    }
}

impl CorkumHighHarmonicAttosecond {
    /// Creates a new Corkum HHG Attosecond engine.
    pub fn new(ionization_potential_ip: f64) -> Result<Self> {
        if ionization_potential_ip <= 0.0 || !ionization_potential_ip.is_finite() {
            return Err(Error::Msg(
                "ionization_potential_ip must be > 0".to_string(),
            ));
        }
        Ok(Self {
            ionization_potential_ip,
        })
    }

    /// Evaluates the HHG cutoff energy $E_{\mathrm{max}} = I_p + 3.17 U_p$, Keldysh parameter $\gamma_K = \sqrt{I_p / (2 U_p)}$, and tunneling regime flag ($\gamma_K < 1$).
    pub fn cutoff_energy_and_keldysh(&self, ponderomotive_up: f64) -> Result<(f64, f64, bool)> {
        if ponderomotive_up <= 0.0 {
            return Err(Error::Msg("ponderomotive_up must be > 0".to_string()));
        }
        let e_cutoff = self.ionization_potential_ip + 3.17 * ponderomotive_up;
        let gamma_k = (self.ionization_potential_ip / (2.0 * ponderomotive_up)).sqrt();
        Ok((e_cutoff, gamma_k, gamma_k < 1.0))
    }
}
