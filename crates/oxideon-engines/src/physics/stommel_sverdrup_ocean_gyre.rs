//! Sverdrup Interior Vorticity Balance & Stommel Western Boundary Current Intensification (Sverdrup, 1947; Stommel, 1948).
//!
//! ### 1. Sverdrup Meridional Transport & Stommel Boundary Layer Thickness $\delta_S$
//! Under planetary vorticity gradient $\beta = \partial f / \partial y$ and linear bottom drag $r$,
//! the western boundary current thickness is $\delta_S = r / \beta$, and the Sverdrup meridional mass transport is $M_y = \frac{1}{\beta} (\nabla \times \boldsymbol{\tau})_z$.

use crate::error::{Error, Result};

/// Sverdrup Vorticity Balance & Stommel Western Boundary Layer Engine.
#[derive(Debug, Clone, Copy)]
pub struct StommelSverdrupOceanCirculation {
    /// Rossby planetary vorticity gradient $\beta > 0$ ($\mathrm{m^{-1} s^{-1}}$).
    pub beta: f64,
    /// Stommel bottom Rayleigh drag coefficient $r > 0$ ($\mathrm{s^{-1}}$).
    pub drag_r: f64,
}

impl Default for StommelSverdrupOceanCirculation {
    fn default() -> Self {
        Self {
            beta: 2.0e-11,
            drag_r: 1.0e-6,
        }
    }
}

impl StommelSverdrupOceanCirculation {
    /// Creates a new Stommel-Sverdrup Ocean Circulation engine.
    pub fn new(beta: f64, drag_r: f64) -> Result<Self> {
        if beta <= 0.0 || drag_r <= 0.0 {
            return Err(Error::Msg("beta and drag_r must be > 0".to_string()));
        }
        Ok(Self { beta, drag_r })
    }

    /// Evaluates the Stommel western boundary width $\delta_S = r / \beta$, Munk viscous width $\delta_M = (A_H / \beta)^{1/3}$, and Sverdrup transport $V_y = \mathrm{curl}_z(\tau) / (\rho_0 \beta)$.
    pub fn boundary_widths_and_sverdrup_transport(
        &self,
        wind_curl_pa_m: f64,
        rho0: f64,
        viscosity_ah: f64,
    ) -> Result<(f64, f64, f64)> {
        if rho0 <= 0.0 || viscosity_ah <= 0.0 {
            return Err(Error::Msg("rho0 and viscosity_ah must be > 0".to_string()));
        }
        let delta_s = self.drag_r / self.beta;
        let delta_m = (viscosity_ah / self.beta).cbrt();
        let v_y = wind_curl_pa_m / (rho0 * self.beta);
        Ok((delta_s, delta_m, v_y))
    }
}
