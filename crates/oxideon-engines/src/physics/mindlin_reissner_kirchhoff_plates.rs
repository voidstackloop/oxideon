//! Kirchhoff-Love Biharmonic Plate Theory $D \nabla^4 w = q$ & Mindlin-Reissner Shear-Deformable Plates (Kirchhoff, 1850; Mindlin, 1951).
//!
//! ### 1. Flexural Rigidity $D$ & Navier Simply-Supported Rectangular Plate Eigenfrequencies
//! A thin elastic plate of thickness $h$, Young's modulus $E$, and Poisson ratio $\nu$ has flexural rigidity:
//! $$D = \frac{E h^3}{12 (1 - \nu^2)}$$
//! Under sinusoidal transverse load $q(x, y) = q_0 \sin(\frac{\pi x}{a}) \sin(\frac{\pi y}{b})$ on $[0, a] \times [0, b]$,
//! the exact Navier deflection amplitude is $w_{\max} = \frac{q_0}{D \pi^4 (a^{-2} + b^{-2})^2}$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Kirchhoff-Love Biharmonic Plate & Flexural Rigidity Engine.
#[derive(Debug, Clone, Copy)]
pub struct KirchhoffLovePlateEngine {
    /// Young's modulus $E > 0$.
    pub young_e: f64,
    /// Plate thickness $h > 0$.
    pub thickness_h: f64,
    /// Poisson's ratio $\nu \in [0, 0.5)$.
    pub poisson_nu: f64,
}

impl KirchhoffLovePlateEngine {
    /// Creates a new Kirchhoff-Love Plate engine.
    pub fn new(young_e: f64, thickness_h: f64, poisson_nu: f64) -> Result<Self> {
        if young_e <= 0.0 || thickness_h <= 0.0 || !(0.0..0.5).contains(&poisson_nu) {
            return Err(Error::Msg(
                "young_e > 0, thickness_h > 0, and poisson_nu in [0, 0.5) required".to_string(),
            ));
        }
        Ok(Self {
            young_e,
            thickness_h,
            poisson_nu,
        })
    }

    /// Computes the plate flexural rigidity $D = \frac{E h^3}{12(1 - \nu^2)}$.
    pub fn flexural_rigidity_d(&self) -> f64 {
        self.young_e * self.thickness_h.powi(3) / (12.0 * (1.0 - self.poisson_nu * self.poisson_nu))
    }

    /// Evaluates Navier's exact central deflection $w_{\max}$ of a simply-supported square plate of side $a > 0$ under sinusoidal pressure $q_0$.
    pub fn navier_square_plate_deflection(&self, side_a: f64, load_q0: f64) -> Result<f64> {
        if side_a <= 0.0 {
            return Err(Error::Msg("side_a must be > 0".to_string()));
        }
        let d = self.flexural_rigidity_d();
        Ok(load_q0 * side_a.powi(4) / (4.0 * PI.powi(4) * d))
    }
}
