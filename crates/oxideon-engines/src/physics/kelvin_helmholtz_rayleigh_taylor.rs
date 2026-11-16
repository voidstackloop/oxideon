//! Kelvin-Helmholtz Shear Instability, Rayleigh-Taylor Buoyancy Instability & Capillary Cutoff Wavenumber.
//!
//! ### 1. Unified Interfacial Kelvin-Helmholtz / Rayleigh-Taylor Dispersion Relation
//! At a horizontal interface between lower fluid $\rho_1$ and upper fluid $\rho_2$ with velocity jump $\Delta U$,
//! gravitational acceleration $g$, and surface tension $\sigma \ge 0$, the growth rate Squared $\gamma^2(k)$ is:
//! $$\gamma^2(k) = \frac{\rho_1 \rho_2}{(\rho_1 + \rho_2)^2} k^2 (\Delta U)^2 + \frac{(\rho_2 - \rho_1) g k - \sigma k^3}{\rho_1 + \rho_2}$$

use crate::error::{Error, Result};

/// Kelvin-Helmholtz & Rayleigh-Taylor Interfacial Instability Engine.
#[derive(Debug, Clone, Copy)]
pub struct HydrodynamicInstabilityEngine {
    /// Lower fluid density $\rho_1 > 0$.
    pub rho1: f64,
    /// Upper fluid density $\rho_2 > 0$.
    pub rho2: f64,
    /// Surface tension $\sigma \ge 0$.
    pub surface_tension_sigma: f64,
}

impl HydrodynamicInstabilityEngine {
    /// Creates a new Hydrodynamic Instability engine.
    pub fn new(rho1: f64, rho2: f64, surface_tension_sigma: f64) -> Result<Self> {
        if rho1 <= 0.0 || rho2 <= 0.0 || surface_tension_sigma < 0.0 {
            return Err(Error::Msg(
                "Fluid densities must be > 0 and surface tension >= 0".to_string(),
            ));
        }
        Ok(Self {
            rho1,
            rho2,
            surface_tension_sigma,
        })
    }

    /// Computes the dimensionless Atwood number $A = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} \in (-1, 1)$.
    pub fn atwood_number(&self) -> f64 {
        (self.rho2 - self.rho1) / (self.rho2 + self.rho1)
    }

    /// Computes the squared instability growth rate $\gamma^2(k)$ (> 0 indicates exponential KH/RT growth, < 0 indicates stable capillary-gravity waves).
    pub fn growth_rate_squared(
        &self,
        wavenumber_k: f64,
        shear_delta_u: f64,
        gravity_g: f64,
    ) -> f64 {
        let sum_rho = self.rho1 + self.rho2;
        let kh_term = (self.rho1 * self.rho2 / (sum_rho * sum_rho))
            * wavenumber_k
            * wavenumber_k
            * shear_delta_u
            * shear_delta_u;
        let rt_term = ((self.rho2 - self.rho1) * gravity_g * wavenumber_k
            - self.surface_tension_sigma * wavenumber_k.powi(3))
            / sum_rho;
        kh_term + rt_term
    }
}
