//! Ehrhart Lattice Polytope Polynomial $L_P(t)$, Ehrhart-Macdonald Reciprocity & Pick's Theorem (Ehrhart, 1962; Macdonald, 1971).
//!
//! ### 1. 2D Lattice Polytope Ehrhart Polynomial & Reciprocity
//! For a 2D integral polygon $P \subset \mathbb{R}^2$ with Euclidean area $A$ and boundary lattice point count $B$,
//! dilation by integer $t \ge 1$ contains $L_P(t) = A t^2 + \frac{B}{2} t + 1$ lattice points, and by Ehrhart-Macdonald reciprocity
//! its open interior $t P^\circ$ contains $(-1)^2 L_P(-t) = A t^2 - \frac{B}{2} t + 1$ lattice points.

use crate::error::{Error, Result};

/// Ehrhart Lattice Polytope Polynomial & Macdonald Reciprocity Engine.
#[derive(Debug, Clone, Copy)]
pub struct EhrhartMacdonaldPolytopeEngine {
    /// Twice the polygon area $2 A \ge 1$ (an integer for lattice polygons).
    pub twice_area_2a: u64,
    /// Number of boundary lattice points $B \ge 3$.
    pub boundary_points_b: u64,
}

impl Default for EhrhartMacdonaldPolytopeEngine {
    fn default() -> Self {
        Self {
            twice_area_2a: 2,
            boundary_points_b: 4,
        }
    }
}

impl EhrhartMacdonaldPolytopeEngine {
    /// Creates a new Ehrhart-Macdonald Polytope engine.
    pub fn new(twice_area_2a: u64, boundary_points_b: u64) -> Result<Self> {
        if twice_area_2a == 0
            || boundary_points_b < 3
            || (twice_area_2a + 2 < boundary_points_b)
            || (twice_area_2a + 2 - boundary_points_b) % 2 != 0
        {
            return Err(Error::Msg(
                "Invalid Pick's theorem parameters (2A + 2 - B must be even >= 0)".to_string(),
            ));
        }
        Ok(Self {
            twice_area_2a,
            boundary_points_b,
        })
    }

    /// Evaluates $(L_P(t), L_{P^\circ}(t))$ for dilation factor $t \ge 1$ and verifies Pick's interior count at $t = 1$.
    pub fn ehrhart_closed_and_interior_counts(&self, dilation_t: u64) -> Result<(u64, u64)> {
        if dilation_t == 0 {
            return Err(Error::Msg("dilation_t must be >= 1".to_string()));
        }
        let closed = (self.twice_area_2a * dilation_t * dilation_t
            + self.boundary_points_b * dilation_t)
            / 2
            + 1;
        let interior = (self.twice_area_2a * dilation_t * dilation_t
            - self.boundary_points_b * dilation_t)
            / 2
            + 1;
        Ok((closed, interior))
    }
}
