//! Tolman-Oppenheimer-Volkoff (TOV) Relativistic Neutron Star Equilibrium, Buchdahl Bound $M/R < 4/9$ & Tidal Deformability $\Lambda$ (Oppenheimer & Volkoff, 1939).
//!
//! ### 1. Relativistic Hydrostatic Pressure Gradient & Buchdahl's Theorem (1959)
//! In general relativity ($G = c = 1$), any static spherically symmetric perfect fluid sphere satisfies Buchdahl's absolute
//! compactness limit $\mathcal{C} = \frac{M}{R} < \frac{4}{9}$, and the dimensionless gravitational-wave **Tidal Deformability** is:
//! $$\Lambda = \frac{2}{3} k_2 \left(\frac{R}{M}\right)^5$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Tolman-Oppenheimer-Volkoff (TOV) Neutron Star & Tidal Love Number Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct TolmanOppenheimerVolkoffEngine;

impl TolmanOppenheimerVolkoffEngine {
    /// Creates a new TOV Neutron Star engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the relativistic TOV pressure gradient $dP / dr$ ($G = c = 1$) at radius $r > 2 m(r)$.
    pub fn tov_pressure_gradient(
        r: f64,
        mass_m: f64,
        density_rho: f64,
        pressure_p: f64,
    ) -> Result<f64> {
        if r <= 2.0 * mass_m || r <= 0.0 {
            return Err(Error::Msg(
                "Radius r must exceed Schwarzschild radius 2*m(r)".to_string(),
            ));
        }
        let num = -(density_rho + pressure_p) * (mass_m + 4.0 * PI * r.powi(3) * pressure_p);
        let den = r * (r - 2.0 * mass_m);
        Ok(num / den)
    }

    /// Computes the dimensionless tidal deformability $\Lambda = \frac{2}{3} k_2 \mathcal{C}^{-5}$ and checks Buchdahl's bound $\mathcal{C} = M/R < 4/9$.
    pub fn tidal_deformability_and_buchdahl_ok(
        compactness_c: f64,
        love_number_k2: f64,
    ) -> Result<(f64, bool)> {
        if compactness_c <= 0.0 || love_number_k2 < 0.0 {
            return Err(Error::Msg(
                "compactness_c > 0 and love_number_k2 >= 0 required".to_string(),
            ));
        }
        let lambda = (2.0 / 3.0) * love_number_k2 / compactness_c.powi(5);
        Ok((lambda, compactness_c < 4.0 / 9.0))
    }
}
