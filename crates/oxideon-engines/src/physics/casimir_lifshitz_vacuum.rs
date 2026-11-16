//! Casimir-Polder Vacuum Zero-Point Force & Lifshitz Dielectric Slab Theory (Casimir, 1948; Lifshitz, 1956).
//!
//! ### 1. Casimir Vacuum Energy Density and Pressure Between Parallel Conductors
//! In natural units ($\hbar = c = 1$), the renormalized vacuum energy per unit area and attractive pressure at gap $a > 0$ are:
//! $$\frac{\mathcal{E}_C(a)}{A} = -\frac{\pi^2}{720 a^3}, \qquad P_C(a) = -\frac{\pi^2}{240 a^4}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Casimir-Lifshitz Vacuum Zero-Point Pressure Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CasimirLifshitzVacuumEngine;

impl CasimirLifshitzVacuumEngine {
    /// Creates a new Casimir-Lifshitz Vacuum engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Casimir energy per unit area $\mathcal{E}/A$ and attractive pressure $P_C(a)$ at separation $a > 0$ ($\hbar = c = 1$).
    pub fn energy_and_pressure(gap_a: f64) -> Result<(f64, f64)> {
        if gap_a <= 0.0 || !gap_a.is_finite() {
            return Err(Error::Msg("Gap separation a must be > 0".to_string()));
        }
        let pi2 = PI * PI;
        let energy_per_area = -pi2 / (720.0 * gap_a.powi(3));
        let pressure = -pi2 / (240.0 * gap_a.powi(4));
        Ok((energy_per_area, pressure))
    }
}
