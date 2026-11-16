//! Strichartz Dispersive Space-Time Estimates for Schrödinger & Wave Equations (Strichartz, 1977; Keel & Tao, 1998).
//!
//! ### 1. Schrödinger Admissible Exponents $(q, r)$ in $\mathbb{R}^d$
//! For the free Schrödinger propagator $e^{i t \Delta}$ on $\mathbb{R}^d$, the space-time norm $\|e^{i t \Delta} u_0\|_{L_t^q L_x^r} \le C \|u_0\|_{L_x^2}$
//! holds iff $(q, r)$ is Schrödinger-admissible ($q, r \ge 2$, $(q, r, d) \ne (2, \infty, 2)$):
//! $$\frac{2}{q} + \frac{d}{r} = \frac{d}{2}$$

use crate::error::{Error, Result};

/// Strichartz Dispersive Space-Time Admissibility & Scaling Engine.
#[derive(Debug, Clone, Copy)]
pub struct StrichartzDispersiveEstimate {
    /// Spatial dimension $d \ge 1$.
    pub dim_d: usize,
}

impl Default for StrichartzDispersiveEstimate {
    fn default() -> Self {
        Self { dim_d: 3 }
    }
}

impl StrichartzDispersiveEstimate {
    /// Creates a new Strichartz Dispersive Estimate engine.
    pub fn new(dim_d: usize) -> Result<Self> {
        if dim_d == 0 {
            return Err(Error::Msg("dim_d must be >= 1".to_string()));
        }
        Ok(Self { dim_d })
    }

    /// Computes the conjugate spatial exponent $r$ such that $2/q + d/r = d/2$ for given temporal exponent $q \ge 2$, and returns the symmetric diagonal exponent $q_{\mathrm{sym}} = r_{\mathrm{sym}} = 2(d + 2)/d$.
    pub fn admissible_spatial_exponent_and_diagonal(
        &self,
        time_q: f64,
    ) -> Result<(f64, f64, bool)> {
        if time_q < 2.0 {
            return Err(Error::Msg("time_q must be >= 2.0".to_string()));
        }
        let d = self.dim_d as f64;
        let rhs = 0.5 * d - 2.0 / time_q;
        if rhs <= 0.0 {
            return Err(Error::Msg(
                "No finite spatial exponent r exists for given (d, q)".to_string(),
            ));
        }
        let r = d / rhs;
        let q_sym = 2.0 * (d + 2.0) / d;
        let valid = r >= 2.0 && !(self.dim_d == 2 && (time_q - 2.0).abs() < 1e-12);
        Ok((r, q_sym, valid))
    }
}
