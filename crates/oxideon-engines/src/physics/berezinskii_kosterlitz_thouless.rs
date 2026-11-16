//! Berezinskii-Kosterlitz-Thouless (BKT) 2D Vortex Unbinding Transition & Universal Stiffness Jump (Kosterlitz & Thouless, 1973).
//!
//! ### 1. Vortex Free Energy $F_v = (\pi J_R - 2 k_B T) \ln(L/a)$ & Nelson-Kosterlitz Universal Jump
//! In the 2D XY model / 2D superfluid film, logarithmic vortex-antivortex binding breaks down at $k_B T_{\mathrm{BKT}} = \frac{\pi}{2} J_R(T_{\mathrm{BKT}}^-)$,
//! yielding the universal **Nelson-Kosterlitz Superfluid Stiffness Jump**:
//! $$\frac{\rho_s(T_{\mathrm{BKT}}^-)}{k_B T_{\mathrm{BKT}}} = \frac{2}{\pi}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Berezinskii-Kosterlitz-Thouless (BKT) 2D Vortex Unbinding & RG Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct BktVortexUnbindingEngine;

impl BktVortexUnbindingEngine {
    /// Creates a new BKT Vortex Unbinding engine.
    pub fn new() -> Self {
        Self
    }

    /// Returns the exact Nelson-Kosterlitz universal superfluid stiffness jump $2 / \pi \approx 0.63661977$.
    pub fn universal_nelson_kosterlitz_jump() -> f64 {
        2.0 / PI
    }

    /// Evaluates the Kosterlitz RG correlation length $\xi(T) = a \exp\left(b / \sqrt{T - T_{\mathrm{BKT}}}\right)$ above $T_{\mathrm{BKT}}$.
    pub fn bkt_correlation_length(
        temp: f64,
        temp_bkt: f64,
        lattice_a: f64,
        scale_b: f64,
    ) -> Result<f64> {
        if temp <= temp_bkt || lattice_a <= 0.0 || scale_b <= 0.0 {
            return Err(Error::Msg(
                "Requires temp > temp_bkt, lattice_a > 0, and scale_b > 0".to_string(),
            ));
        }
        Ok(lattice_a * (scale_b / (temp - temp_bkt).sqrt()).exp())
    }
}
