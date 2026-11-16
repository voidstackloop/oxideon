//! Mikhalkin Tropical Curve Correspondence Theorem, Tropical Semiring $(\mathbb{R}, \oplus, \odot)$ & Newton Polygon Genus (Mikhalkin, 2005).
//!
//! ### 1. Tropical Polynomial Evaluation & Smooth Degree-$d$ Tropical Plane Curve Genus
//! Over the min-plus tropical semiring ($a \oplus b = \min(a, b)$, $a \odot b = a + b$), a smooth tropical curve of degree $d$ in $\mathbb{TP}^2$
//! dual to a unimodular triangulation of $\Delta_d$ has genus $g = \binom{d - 1}{2}$ and intersects a degree-$e$ curve in $d \cdot e$ points counted with determinant multiplicity.

use crate::error::{Error, Result};

/// Mikhalkin Tropical Curve & Min-Plus Semiring Engine.
#[derive(Debug, Clone, Copy)]
pub struct MikhalkinTropicalGeometryEngine {
    /// Tropical plane curve degree $d \ge 1$.
    pub degree_d: usize,
}

impl Default for MikhalkinTropicalGeometryEngine {
    fn default() -> Self {
        Self { degree_d: 3 }
    }
}

impl MikhalkinTropicalGeometryEngine {
    /// Creates a new Mikhalkin Tropical Geometry engine.
    pub fn new(degree_d: usize) -> Result<Self> {
        if degree_d == 0 {
            return Err(Error::Msg("degree_d must be >= 1".to_string()));
        }
        Ok(Self { degree_d })
    }

    /// Evaluates the interior lattice point genus $g = (d-1)(d-2)/2$, tropical Bézout intersection count $d \cdot e$, and univariate min-plus tropical polynomial $\min_k (c_k + k x)$.
    pub fn genus_bezout_and_tropical_eval(
        &self,
        other_degree_e: usize,
        coeffs: &[f64],
        x: f64,
    ) -> Result<(usize, usize, f64)> {
        if other_degree_e == 0 || coeffs.is_empty() {
            return Err(Error::Msg(
                "other_degree_e >= 1 and non-empty coeffs required".to_string(),
            ));
        }
        let genus = (self.degree_d - 1) * self.degree_d.saturating_sub(2) / 2;
        let bezout = self.degree_d * other_degree_e;
        let mut trop_val = f64::INFINITY;
        for (k, &c) in coeffs.iter().enumerate() {
            let term = c + (k as f64) * x;
            if term < trop_val {
                trop_val = term;
            }
        }
        Ok((genus, bezout, trop_val))
    }
}
