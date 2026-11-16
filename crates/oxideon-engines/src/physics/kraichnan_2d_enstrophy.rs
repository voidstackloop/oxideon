//! Kraichnan-Batchelor 2D Turbulence Dual Cascade: Inverse Energy $k^{-5/3}$ & Forward Enstrophy $k^{-3}$ (Kraichnan, 1967).
//!
//! ### 1. Dual Conservation of Energy $E$ & Enstrophy $Z = \frac{1}{2} \int \omega^2 d^2x$
//! In 2D inviscid incompressible flow, vorticity $\omega = \nabla \times \mathbf{u}$ is materially conserved, forcing
//! injected energy at wavenumber $k_f$ into an **Inverse Energy Cascade** ($E(k) = C_E \varepsilon^{2/3} k^{-5/3}$ for $k < k_f$)
//! and injected enstrophy at rate $\eta_Z$ into a **Forward Enstrophy Cascade** ($E(k) = C_Z \eta_Z^{2/3} k^{-3}$ for $k > k_f$).

use crate::error::{Error, Result};

/// Kraichnan-Batchelor 2D Dual Cascade Engine.
#[derive(Debug, Clone, Copy)]
pub struct KraichnanEnstrophyCascade {
    /// Forcing wavenumber $k_f > 0$.
    pub forcing_kf: f64,
    /// Upscale energy flux $\varepsilon > 0$.
    pub energy_flux_eps: f64,
    /// Downscale enstrophy flux $\eta_Z > 0$.
    pub enstrophy_flux_eta: f64,
}

impl KraichnanEnstrophyCascade {
    /// Creates a new Kraichnan 2D Dual Cascade engine.
    pub fn new(forcing_kf: f64, energy_flux_eps: f64, enstrophy_flux_eta: f64) -> Result<Self> {
        if forcing_kf <= 0.0 || energy_flux_eps <= 0.0 || enstrophy_flux_eta <= 0.0 {
            return Err(Error::Msg(
                "All Kraichnan 2D cascade parameters must be > 0".to_string(),
            ));
        }
        Ok(Self {
            forcing_kf,
            energy_flux_eps,
            enstrophy_flux_eta,
        })
    }

    /// Evaluates the piecewise dual-cascade energy spectrum $E(k)$ ($k^{-5/3}$ for $k \le k_f$, $k^{-3}$ for $k > k_f$).
    pub fn dual_cascade_spectrum(&self, k: f64) -> Result<f64> {
        if k <= 0.0 {
            return Err(Error::Msg("Wavenumber k must be > 0".to_string()));
        }
        if k <= self.forcing_kf {
            Ok(6.0 * self.energy_flux_eps.powf(2.0 / 3.0) * k.powf(-5.0 / 3.0))
        } else {
            Ok(1.5 * self.enstrophy_flux_eta.powf(2.0 / 3.0) * k.powi(-3))
        }
    }
}
