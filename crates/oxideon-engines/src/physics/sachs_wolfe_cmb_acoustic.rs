//! Cosmic Microwave Background (CMB) Baryon-Photon Acoustic Oscillations & Sachs-Wolfe Effect (Sachs & Wolfe, 1967; Hu & Sugiyama, 1995).
//!
//! ### 1. Baryon-Photon Fluid Sound Speed $c_s$ & Acoustic Peak Multipoles $\ell_m$
//! Prior to recombination ($z_* \approx 1090$), tight Thomson coupling binds photons and baryons into a single fluid
//! with momentum density ratio $R_b = \frac{3 \rho_b}{4 \rho_\gamma}$ and relativistic acoustic sound speed:
//! $$\frac{c_s}{c} = \frac{1}{\sqrt{3 (1 + R_b)}}$$

use crate::error::{Error, Result};

/// CMB Baryon-Photon Acoustic Sound Horizon & Sachs-Wolfe Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CmbAcousticPeakEngine;

impl CmbAcousticPeakEngine {
    /// Creates a new CMB Acoustic Peak engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the dimensionless baryon-photon sound speed $c_s / c = 1 / \sqrt{3 (1 + R_b)}$ for baryon loading $R_b = \frac{3 \rho_b}{4 \rho_\gamma} \ge 0$.
    pub fn baryon_photon_sound_speed_ratio(baryon_loading_rb: f64) -> Result<f64> {
        if baryon_loading_rb < 0.0 {
            return Err(Error::Msg("baryon_loading_rb must be >= 0".to_string()));
        }
        Ok(1.0 / (3.0 * (1.0 + baryon_loading_rb)).sqrt())
    }
}
