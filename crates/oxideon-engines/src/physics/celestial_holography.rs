//! Celestial Holography, Mellin Boost Conformal Primary Wavefunctions & BMS Soft Graviton Theorems (Pasterski, Shao & Strominger, 2017).
//!
//! ### 1. Pasterski-Shao-Strominger Mellin Transform
//! Scattering amplitudes of massless particles with energy $\omega$ are mapped to 2D Celestial CFT correlators
//! on the celestial sphere $\mathcal{CS}^2$ of conformal dimension $\Delta = 1 + i \lambda$ via the Mellin transform:
//! $$\tilde{\mathcal{A}}(\Delta_i, z_i, \bar{z}_i) = \left(\prod_i \int_0^\infty d\omega_i \, \omega_i^{\Delta_i - 1}\right) \mathcal{A}(\omega_i, z_i, \bar{z}_i)$$
//! For an exponentially regulated UV amplitude $e^{-s \omega}$, the Mellin transform yields $\Gamma(\Delta) s^{-\Delta}$.

use crate::error::{Error, Result};

/// Celestial Holography Mellin Transform & BMS Soft Theorem Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CelestialHolographyEngine;

impl CelestialHolographyEngine {
    /// Creates a new Celestial Holography engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the exact Mellin-transformed celestial amplitude $\int_0^\infty \omega^{\Delta - 1} e^{-s \omega} d\omega = \Gamma(\Delta) s^{-\Delta}$
    /// for integer conformal weight $\Delta \in 1..=10$ and UV scale $s > 0$.
    pub fn mellin_celestial_correlator(delta: u32, uv_scale_s: f64) -> Result<f64> {
        if !(1..=10).contains(&delta) || uv_scale_s <= 0.0 {
            return Err(Error::Msg(
                "delta must be in 1..=10 and uv_scale_s > 0".to_string(),
            ));
        }
        let mut gamma_val = 1.0;
        for k in 1..delta {
            gamma_val *= k as f64;
        }
        Ok(gamma_val * uv_scale_s.powi(-(delta as i32)))
    }

    /// Evaluates Weinberg's soft photon/graviton pole residue $\lim_{\Delta \to 1} (\Delta - 1) \mathcal{M}(\Delta, s)$
    /// where $\mathcal{M}(\Delta, s) = \frac{s^{1 - \Delta}}{\Delta - 1}$.
    pub fn weinberg_soft_pole_residue(delta: f64, scale_s: f64) -> Result<f64> {
        if scale_s <= 0.0 {
            return Err(Error::Msg("scale_s must be > 0".to_string()));
        }
        Ok(scale_s.powf(1.0 - delta))
    }
}
