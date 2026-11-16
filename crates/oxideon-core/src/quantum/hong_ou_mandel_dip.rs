//! Hong-Ou-Mandel (HOM) Two-Photon Quantum Interference Dip & Bosonic Bunching (Hong, Ou & Mandel, 1987).
//!
//! ### 1. Coincidence Probability Across a Beam Splitter with Reflectance $R$ and Transmittance $T = 1 - R$
//! For two photons with Gaussian spectral bandwidth $\sigma_\omega$, temporal delay $\Delta \tau$, and internal state overlap $|\langle \psi_1 | \psi_2 \rangle|^2 = V \in [0, 1]$:
//! $$P_{\mathrm{coinc}}(\Delta \tau) = T^2 + R^2 - 2 T R V \exp(-\sigma_\omega^2 \Delta \tau^2)$$

use crate::error::{Error, Result};

/// Hong-Ou-Mandel Two-Photon Quantum Interference Engine.
#[derive(Debug, Clone, Copy)]
pub struct HongOuMandelInterferometer {
    /// Beam splitter reflectance $R \in (0, 1)$ ($R = 0.5$ for balanced 50:50).
    pub reflectance_r: f64,
    /// Spectral bandwidth $\sigma_\omega > 0$.
    pub bandwidth_sigma: f64,
}

impl Default for HongOuMandelInterferometer {
    fn default() -> Self {
        Self {
            reflectance_r: 0.5,
            bandwidth_sigma: 1.0,
        }
    }
}

impl HongOuMandelInterferometer {
    /// Creates a new Hong-Ou-Mandel Interferometer engine.
    pub fn new(reflectance_r: f64, bandwidth_sigma: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&reflectance_r) || reflectance_r == 0.0 || bandwidth_sigma <= 0.0 {
            return Err(Error::Msg(
                "reflectance_r in (0, 1) and bandwidth_sigma > 0 required".to_string(),
            ));
        }
        Ok(Self {
            reflectance_r,
            bandwidth_sigma,
        })
    }

    /// Evaluates the two-photon coincidence probability $P_{\mathrm{coinc}}(\Delta \tau)$ and HOM dip visibility for state overlap $V \in [0, 1]$.
    pub fn coincidence_probability_and_visibility(
        &self,
        delay_tau: f64,
        state_overlap_v: f64,
    ) -> Result<(f64, f64)> {
        if !(0.0..=1.0).contains(&state_overlap_v) {
            return Err(Error::Msg("state_overlap_v must be in [0, 1]".to_string()));
        }
        let r = self.reflectance_r;
        let t = 1.0 - r;
        let p_inf = t * t + r * r;
        let dip =
            2.0 * t * r * state_overlap_v * (-(self.bandwidth_sigma * delay_tau).powi(2)).exp();
        let vis = (2.0 * t * r * state_overlap_v) / p_inf;
        Ok((p_inf - dip, vis))
    }
}
