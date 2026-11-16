//! Collisionless Vlasov-Poisson Plasma Kinetics, Bohm-Gross Dispersion & Landau Damping Rate (Landau, 1946).
//!
//! ### 1. Bohm-Gross Langmuir Frequency $\omega_r(k)$ & Collisionless Landau Damping $\gamma_L(k)$
//! In a collisionless Maxwellian electron plasma with plasma frequency $\omega_{pe}$ and Debye length $\lambda_D = v_{\text{th}} / \omega_{pe}$,
//! long-wavelength Langmuir waves ($k \lambda_D \ll 1$) satisfy:
//! $$\omega_r(k) = \omega_{pe} \sqrt{1 + 3 (k \lambda_D)^2}, \qquad \gamma_L(k) = -\sqrt{\frac{\pi}{8}} \frac{\omega_{pe}}{(k \lambda_D)^3} \exp\left(-\frac{1}{2 (k \lambda_D)^2} - \frac{3}{2}\right) < 0$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Collisionless Vlasov-Poisson Plasma & Landau Damping Engine.
#[derive(Debug, Clone, Copy)]
pub struct VlasovLandauDampingEngine {
    /// Electron plasma frequency $\omega_{pe} > 0$.
    pub omega_pe: f64,
    /// Debye screening length $\lambda_D > 0$.
    pub debye_length: f64,
}

impl VlasovLandauDampingEngine {
    /// Creates a new Vlasov-Landau plasma kinetics engine.
    pub fn new(omega_pe: f64, debye_length: f64) -> Result<Self> {
        if omega_pe <= 0.0 || debye_length <= 0.0 {
            return Err(Error::Msg(
                "omega_pe and debye_length must be > 0".to_string(),
            ));
        }
        Ok(Self {
            omega_pe,
            debye_length,
        })
    }

    /// Computes the Bohm-Gross real frequency $\omega_r(k)$ and Landau damping rate $\gamma_L(k) < 0$.
    pub fn bohm_gross_and_landau_damping(&self, wavenumber_k: f64) -> Result<(f64, f64)> {
        if wavenumber_k <= 0.0 {
            return Err(Error::Msg("wavenumber_k must be > 0".to_string()));
        }
        let kl = wavenumber_k * self.debye_length;
        let omega_r = self.omega_pe * (1.0 + 3.0 * kl * kl).sqrt();
        let gamma_l =
            -(PI / 8.0).sqrt() * self.omega_pe / kl.powi(3) * (-0.5 / (kl * kl) - 1.5).exp();
        Ok((omega_r, gamma_l))
    }
}
