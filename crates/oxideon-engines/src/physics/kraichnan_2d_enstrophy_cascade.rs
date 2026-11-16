//! Kraichnan-Batchelor 2D Turbulence Dual Cascade: Inverse Energy $k^{-5/3}$ & Forward Enstrophy $k^{-3}$ (Kraichnan, 1967; Batchelor, 1969).
//!
//! ### 1. Enstrophy Flux $\eta_\omega$ & Log-Corrected Enstrophy Spectrum $E(k) = C' \eta_\omega^{2/3} k^{-3} [\ln(k / k_f)]^{-1/3}$
//! In two-dimensional incompressible Navier-Stokes with forcing at wavenumber $k_f$:
//! - For $k < k_f$ (inverse energy cascade): $E(k) = C_E \varepsilon^{2/3} k^{-5/3}$
//! - For $k > k_f$ (forward enstrophy cascade): $E(k) = C_\omega \eta_\omega^{2/3} k^{-3} [\ln(k / k_f)]^{-1/3}$

use crate::error::{Error, Result};

/// Kraichnan-Batchelor 2D Turbulence Dual Cascade Engine.
#[derive(Debug, Clone, Copy)]
pub struct Kraichnan2DTurbulenceCascade {
    /// Forcing wavenumber $k_f > 0$.
    pub forcing_kf: f64,
    /// Enstrophy transfer rate $\eta_\omega > 0$.
    pub enstrophy_flux_eta: f64,
}

impl Default for Kraichnan2DTurbulenceCascade {
    fn default() -> Self {
        Self {
            forcing_kf: 10.0,
            enstrophy_flux_eta: 1.0,
        }
    }
}

impl Kraichnan2DTurbulenceCascade {
    /// Creates a new Kraichnan 2D Turbulence Dual Cascade engine.
    pub fn new(forcing_kf: f64, enstrophy_flux_eta: f64) -> Result<Self> {
        if forcing_kf <= 0.0 || enstrophy_flux_eta <= 0.0 {
            return Err(Error::Msg(
                "forcing_kf and enstrophy_flux_eta must be > 0".to_string(),
            ));
        }
        Ok(Self {
            forcing_kf,
            enstrophy_flux_eta,
        })
    }

    /// Evaluates the Kraichnan log-corrected forward enstrophy spectrum $E(k)$ for $k > k_f$.
    pub fn forward_enstrophy_spectrum(&self, wavenumber_k: f64, c_omega: f64) -> Result<f64> {
        if wavenumber_k <= self.forcing_kf || c_omega <= 0.0 {
            return Err(Error::Msg(
                "wavenumber_k > forcing_kf and c_omega > 0 required".to_string(),
            ));
        }
        let log_factor = (wavenumber_k / self.forcing_kf).ln().powf(-1.0 / 3.0);
        Ok(c_omega * self.enstrophy_flux_eta.powf(2.0 / 3.0) * wavenumber_k.powi(-3) * log_factor)
    }
}
