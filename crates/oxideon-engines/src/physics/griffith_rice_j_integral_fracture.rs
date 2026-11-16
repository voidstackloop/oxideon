//! Griffith Brittle Fracture Criterion, Irwin Stress Intensity Factor $K_I$ & Rice Path-Independent $J$-Integral (Griffith, 1921; Rice, 1968).
//!
//! ### 1. Rice's Path-Independent $J$-Integral & Plane-Strain Energy Release Rate $G$
//! In linear elastic fracture mechanics (LEFM) under plane strain with Young's modulus $E > 0$ and Poisson ratio $\nu \in [0, 0.5)$,
//! Rice's contour-independent $J$-integral equals the Griffith energy release rate:
//! $$J = G = \frac{1 - \nu^2}{E} (K_I^2 + K_{II}^2) + \frac{1 + \nu}{E} K_{III}^2$$

use crate::error::{Error, Result};

/// Griffith-Irwin-Rice Fracture Mechanics & $J$-Integral Engine.
#[derive(Debug, Clone, Copy)]
pub struct RiceJIntegralFractureEngine {
    /// Young's modulus $E > 0$.
    pub young_e: f64,
    /// Poisson's ratio $\nu \in [0, 0.5)$.
    pub poisson_nu: f64,
}

impl RiceJIntegralFractureEngine {
    /// Creates a new Rice $J$-Integral Fracture engine.
    pub fn new(young_e: f64, poisson_nu: f64) -> Result<Self> {
        if young_e <= 0.0 || !(0.0..0.5).contains(&poisson_nu) {
            return Err(Error::Msg(
                "young_e > 0 and poisson_nu in [0, 0.5) required".to_string(),
            ));
        }
        Ok(Self {
            young_e,
            poisson_nu,
        })
    }

    /// Evaluates the plane-strain Rice $J$-integral energy release rate $J(K_I, K_{II}, K_{III})$ from Mode I, II, III stress intensity factors.
    pub fn plane_strain_j_integral(&self, k1: f64, k2: f64, k3: f64) -> f64 {
        let nu = self.poisson_nu;
        (1.0 - nu * nu) / self.young_e * (k1 * k1 + k2 * k2) + (1.0 + nu) / self.young_e * (k3 * k3)
    }
}
