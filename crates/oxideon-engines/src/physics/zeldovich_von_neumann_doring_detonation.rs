//! Zel'dovich-von Neumann-Döring (ZND) Detonation Structure & Chapman-Jouguet Sonic Velocity (Chapman, 1899; Jouguet, 1905; Zel'dovich, 1940).
//!
//! ### 1. Chapman-Jouguet Detonation Velocity $D_{\mathrm{CJ}} = \sqrt{2(\gamma^2 - 1) Q}$ & von Neumann Spike
//! For a reactive gas with specific heat release $Q \gg c_0^2$ and adiabatic index $\gamma > 1$:
//! $$D_{\mathrm{CJ}} = \sqrt{2(\gamma^2 - 1) Q}, \qquad p_{\mathrm{CJ}} = \frac{\rho_0 D_{\mathrm{CJ}}^2}{\gamma + 1}, \qquad p_{\mathrm{vN}} = 2 p_{\mathrm{CJ}}$$

use crate::error::{Error, Result};

/// Zel'dovich-von Neumann-Döring (ZND) & Chapman-Jouguet Detonation Engine.
#[derive(Debug, Clone, Copy)]
pub struct ZeldovichVonNeumannDoringDetonation {
    /// Adiabatic index of detonation products $\gamma > 1$.
    pub gamma_products: f64,
    /// Upstream unburned gas density $\rho_0 > 0$ ($\mathrm{kg/m^3}$).
    pub rho0_kg_m3: f64,
}

impl Default for ZeldovichVonNeumannDoringDetonation {
    fn default() -> Self {
        Self {
            gamma_products: 1.25,
            rho0_kg_m3: 1.2,
        }
    }
}

impl ZeldovichVonNeumannDoringDetonation {
    /// Creates a new ZND / Chapman-Jouguet Detonation engine.
    pub fn new(gamma_products: f64, rho0_kg_m3: f64) -> Result<Self> {
        if gamma_products <= 1.0 || rho0_kg_m3 <= 0.0 {
            return Err(Error::Msg(
                "gamma_products > 1 and rho0_kg_m3 > 0 required".to_string(),
            ));
        }
        Ok(Self {
            gamma_products,
            rho0_kg_m3,
        })
    }

    /// Evaluates the Chapman-Jouguet detonation speed $D_{\mathrm{CJ}}$, CJ pressure $p_{\mathrm{CJ}}$, and von Neumann spike pressure $p_{\mathrm{vN}} = 2 p_{\mathrm{CJ}}$.
    pub fn cj_velocity_and_von_neumann_spike(
        &self,
        heat_release_q: f64,
    ) -> Result<(f64, f64, f64)> {
        if heat_release_q <= 0.0 || !heat_release_q.is_finite() {
            return Err(Error::Msg("heat_release_q must be > 0".to_string()));
        }
        let g = self.gamma_products;
        let d_cj = (2.0 * (g * g - 1.0) * heat_release_q).sqrt();
        let p_cj = self.rho0_kg_m3 * d_cj * d_cj / (g + 1.0);
        Ok((d_cj, p_cj, 2.0 * p_cj))
    }
}
