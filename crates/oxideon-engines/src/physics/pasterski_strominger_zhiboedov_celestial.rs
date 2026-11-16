//! Pasterski-Strominger-Zhiboedov Celestial Holography & Conformal Primary Mellin Transform $\widetilde{\mathcal{A}}(\Delta_i, z_i, \bar{z}_i)$ (Pasterski, Strominger & Zhiboedov, 2017).
//!
//! ### 1. Principal Continuous Series $\Delta = 1 + i \lambda$ & Mellin Energy Transform $\int_0^\infty \omega^{\Delta - 1} e^{-\epsilon \omega} d\omega = \Gamma(\Delta) \epsilon^{-\Delta}$
//! Massless 4D scattering states in momentum basis $(\omega_i, z_i, \bar{z}_i)$ map via Mellin transform in energy $\omega_i$ to 2D celestial CFT primary operators of conformal dimension $\Delta_i = 1 + i \lambda_i$ ($\lambda_i \in \mathbb{R}$) and 2D spin $J_i = \ell_i$ on the celestial sphere $S^2$.

use crate::error::{Error, Result};

/// Pasterski-Strominger-Zhiboedov Celestial Holography Mellin Engine.
#[derive(Debug, Clone, Copy)]
pub struct CelestialHolographyMellinAmplitude {
    /// Ultraviolet regulator $\epsilon > 0$.
    pub uv_regulator_eps: f64,
}

impl Default for CelestialHolographyMellinAmplitude {
    fn default() -> Self {
        Self {
            uv_regulator_eps: 0.5,
        }
    }
}

impl CelestialHolographyMellinAmplitude {
    /// Creates a new Celestial Holography Mellin Amplitude engine.
    pub fn new(uv_regulator_eps: f64) -> Result<Self> {
        if uv_regulator_eps <= 0.0 || !uv_regulator_eps.is_finite() {
            return Err(Error::Msg("uv_regulator_eps must be > 0".to_string()));
        }
        Ok(Self { uv_regulator_eps })
    }

    /// Evaluates the principal-series norm $|\Delta|^2 = 1 + \lambda^2$ and integer-dimension Mellin kernel $(n - 1)! \, \epsilon^{-n}$ for $\Delta = n \ge 1$.
    pub fn principal_norm_sq_and_integer_mellin_kernel(
        &self,
        lambda_real: f64,
        delta_int: u32,
    ) -> Result<(f64, f64)> {
        if delta_int == 0 || delta_int > 12 {
            return Err(Error::Msg("delta_int must be in 1..=12".to_string()));
        }
        let norm_sq = 1.0 + lambda_real * lambda_real;
        let mut gamma_n = 1.0;
        for k in 1..delta_int {
            gamma_n *= k as f64;
        }
        let mellin = gamma_n * self.uv_regulator_eps.powi(-(delta_int as i32));
        Ok((norm_sq, mellin))
    }
}
