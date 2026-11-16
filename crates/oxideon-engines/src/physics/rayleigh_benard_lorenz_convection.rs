//! Rayleigh-Bénard Thermal Convection Threshold ($\mathrm{Ra}_c = 27\pi^4 / 4$) & Lorenz Strange Attractor (Rayleigh, 1916; Lorenz, 1963).
//!
//! ### 1. Free-Free Critical Rayleigh Number & Lorenz Hopf Subcriticality $\rho_H = \sigma(\sigma + b + 3)/(\sigma - b - 1)$
//! For stress-free horizontal boundaries, convection sets in at $k_c = \pi / \sqrt{2}$ and $\mathrm{Ra}_c = 27 \pi^4 / 4 \approx 657.511$.
//! In the truncated Lorenz system ($\dot{X} = \sigma(Y - X)$, $\dot{Y} = \rho X - Y - XZ$, $\dot{Z} = XY - bZ$), the convective fixed points lose stability at $\rho_H$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Rayleigh-Bénard Thermal Convection & Lorenz Attractor Engine.
#[derive(Debug, Clone, Copy)]
pub struct RayleighBenardLorenzConvection {
    /// Prandtl number $\sigma > 0$.
    pub prandtl_sigma: f64,
    /// Geometric aspect parameter $b > 0$ ($8/3$ classically).
    pub aspect_b: f64,
}

impl Default for RayleighBenardLorenzConvection {
    fn default() -> Self {
        Self {
            prandtl_sigma: 10.0,
            aspect_b: 8.0 / 3.0,
        }
    }
}

impl RayleighBenardLorenzConvection {
    /// Creates a new Rayleigh-Bénard & Lorenz Convection engine.
    pub fn new(prandtl_sigma: f64, aspect_b: f64) -> Result<Self> {
        if prandtl_sigma <= aspect_b + 1.0 || aspect_b <= 0.0 {
            return Err(Error::Msg(
                "prandtl_sigma > aspect_b + 1 and aspect_b > 0 required".to_string(),
            ));
        }
        Ok(Self {
            prandtl_sigma,
            aspect_b,
        })
    }

    /// Evaluates the Rayleigh free-free critical threshold $\mathrm{Ra}_c = 27\pi^4/4$ and Lorenz Hopf bifurcation parameter $\rho_H$.
    pub fn critical_rayleigh_and_lorenz_hopf_rho(&self) -> (f64, f64) {
        let ra_c = 27.0 * PI.powi(4) / 4.0;
        let rho_h = self.prandtl_sigma * (self.prandtl_sigma + self.aspect_b + 3.0)
            / (self.prandtl_sigma - self.aspect_b - 1.0);
        (ra_c, rho_h)
    }
}
