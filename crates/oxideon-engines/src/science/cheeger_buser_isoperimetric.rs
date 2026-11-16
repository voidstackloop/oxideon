//! Cheeger Isoperimetric Constant $h(M)$ & Cheeger-Buser Laplacian Spectral Gap Inequalities (Cheeger, 1970; Buser, 1982).
//!
//! ### 1. Cheeger Lower Bound & Buser Upper Bound on $\lambda_1(M)$
//! For a closed Riemannian $n$-manifold $M$ with Cheeger constant $h(M) = \inf_\Omega \frac{\mathrm{Area}(\partial \Omega)}{\min(\mathrm{Vol}(\Omega), \mathrm{Vol}(M \setminus \Omega))}$:
//! $$\frac{1}{4} h(M)^2 \le \lambda_1(M) \le 2 c (n - 1) h(M) + 10 h(M)^2$$

use crate::error::{Error, Result};

/// Cheeger-Buser Isoperimetric & Laplacian Spectral Gap Engine.
#[derive(Debug, Clone, Copy)]
pub struct CheegerBuserSpectralInequality {
    /// Cheeger isoperimetric constant $h(M) > 0$.
    pub cheeger_h: f64,
}

impl Default for CheegerBuserSpectralInequality {
    fn default() -> Self {
        Self { cheeger_h: 2.0 }
    }
}

impl CheegerBuserSpectralInequality {
    /// Creates a new Cheeger-Buser Spectral Inequality engine.
    pub fn new(cheeger_h: f64) -> Result<Self> {
        if cheeger_h <= 0.0 || !cheeger_h.is_finite() {
            return Err(Error::Msg("cheeger_h must be > 0".to_string()));
        }
        Ok(Self { cheeger_h })
    }

    /// Evaluates the Cheeger lower bound $\frac{1}{4} h(M)^2$, zero-Ricci Buser upper bound $10 h(M)^2$, and checks if $\lambda_1$ satisfies Cheeger's inequality.
    pub fn cheeger_lower_bound_and_verify(&self, lambda_1: f64) -> (f64, f64, bool) {
        let lower = 0.25 * self.cheeger_h * self.cheeger_h;
        let buser_upper = 10.0 * self.cheeger_h * self.cheeger_h;
        (lower, buser_upper, lambda_1 >= lower - 1e-12)
    }
}
