//! Magnetohydrodynamics (MHD), Alfvén Waves, Elsässer Variables & Parker-Steenbeck-Krause $\alpha$-$\Omega$ Dynamo (Alfvén, 1942).
//!
//! ### 1. Alfvén Velocity $v_A$ & Mean-Field $\alpha$-$\Omega$ Dynamo Dispersion
//! In a conducting plasma of mass density $\rho > 0$ and magnetic diffusivity $\eta_M > 0$ threaded by field $B_0$:
//! $$v_A = \frac{B_0}{\sqrt{\mu_0 \rho}}, \qquad \gamma_{\text{dyn}}(k) = \sqrt{\frac{1}{2} |\alpha \Omega k|} - \eta_M k^2$$

use crate::error::{Error, Result};

/// Alfvén MHD Wave & Mean-Field $\alpha$-$\Omega$ Dynamo Engine.
#[derive(Debug, Clone, Copy)]
pub struct AlfvenMhdDynamoEngine {
    /// Mass density $\rho > 0$.
    pub density_rho: f64,
    /// Magnetic permeability $\mu_0 > 0$.
    pub mu_0: f64,
    /// Magnetic diffusivity $\eta_M > 0$.
    pub diffusivity_eta: f64,
}

impl AlfvenMhdDynamoEngine {
    /// Creates a new Alfvén MHD Dynamo engine.
    pub fn new(density_rho: f64, mu_0: f64, diffusivity_eta: f64) -> Result<Self> {
        if density_rho <= 0.0 || mu_0 <= 0.0 || diffusivity_eta <= 0.0 {
            return Err(Error::Msg("All MHD parameters must be > 0".to_string()));
        }
        Ok(Self {
            density_rho,
            mu_0,
            diffusivity_eta,
        })
    }

    /// Computes the Alfvén wave phase speed $v_A = |B_0| / \sqrt{\mu_0 \rho}$.
    pub fn alfven_speed(&self, b0: f64) -> f64 {
        b0.abs() / (self.mu_0 * self.density_rho).sqrt()
    }

    /// Computes the mean-field $\alpha$-$\Omega$ dynamo growth rate $\gamma(k) = \sqrt{|\alpha \Omega k| / 2} - \eta_M k^2$.
    pub fn alpha_omega_growth_rate(
        &self,
        alpha_helicity: f64,
        omega_shear: f64,
        wavenumber_k: f64,
    ) -> f64 {
        (0.5 * (alpha_helicity * omega_shear * wavenumber_k).abs()).sqrt()
            - self.diffusivity_eta * wavenumber_k * wavenumber_k
    }
}
