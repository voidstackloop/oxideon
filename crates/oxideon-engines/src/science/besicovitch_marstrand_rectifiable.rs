//! Besicovitch-Federer Structure Theorem & Marstrand Orthogonal Projection / Slicing Theorems (Marstrand, 1954; Federer, 1969).
//!
//! ### 1. Marstrand-Mattila Projection & Slicing Dimension Formulas
//! For a Borel set $E \subset \mathbb{R}^n$ of Hausdorff dimension $s = \dim_{\mathcal{H}}(E)$, for almost every $m$-plane $V \in G(n, m)$:
//! $$\dim_{\mathcal{H}}(\pi_V(E)) = \min(s, m), \qquad \dim_{\mathcal{H}}(E \cap (V^\perp + y)) = \max(0, s - m)$$

use crate::error::{Error, Result};

/// Besicovitch-Federer Rectifiability & Marstrand Projection Engine.
#[derive(Debug, Clone, Copy)]
pub struct BesicovitchMarstrandRectifiable {
    /// Ambient Euclidean dimension $n \ge 2$.
    pub ambient_n: usize,
}

impl Default for BesicovitchMarstrandRectifiable {
    fn default() -> Self {
        Self { ambient_n: 3 }
    }
}

impl BesicovitchMarstrandRectifiable {
    /// Creates a new Besicovitch-Marstrand Rectifiable engine.
    pub fn new(ambient_n: usize) -> Result<Self> {
        if ambient_n < 2 {
            return Err(Error::Msg("ambient_n must be >= 2".to_string()));
        }
        Ok(Self { ambient_n })
    }

    /// Evaluates the generic Marstrand projection dimension $\min(s, m)$, generic slice dimension $\max(0, s - m)$, and integer rectifiability candidate flag.
    pub fn projection_and_slice_dimensions(
        &self,
        hausdorff_dim_s: f64,
        subspace_dim_m: usize,
    ) -> Result<(f64, f64, bool)> {
        if !(0.0..=(self.ambient_n as f64)).contains(&hausdorff_dim_s)
            || subspace_dim_m == 0
            || subspace_dim_m >= self.ambient_n
        {
            return Err(Error::Msg(
                "Invalid Hausdorff dimension s or subspace_dim_m".to_string(),
            ));
        }
        let m = subspace_dim_m as f64;
        let proj_dim = hausdorff_dim_s.min(m);
        let slice_dim = (hausdorff_dim_s - m).max(0.0);
        let is_integer_dim = (hausdorff_dim_s - hausdorff_dim_s.round()).abs() < 1e-12;
        Ok((proj_dim, slice_dim, is_integer_dim))
    }
}
