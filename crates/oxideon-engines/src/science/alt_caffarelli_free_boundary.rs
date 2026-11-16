//! Alt-Caffarelli One-Phase Free-Boundary Problem & Weiss Monotonicity Formula (Alt & Caffarelli, 1981; Weiss, 1999).
//!
//! ### 1. Bernoulli Free-Boundary Condition $|\nabla u| = \sqrt{\Lambda}$ & Weiss Functional $W(r)$
//! Minimizers of the Alt-Caffarelli functional $J(u) = \int (|\nabla u|^2 + \Lambda \mathbf{1}_{\{u > 0\}}) dx$ satisfy $\Delta u = 0$ in $\{u > 0\}$
//! and $|\nabla u| = \sqrt{\Lambda}$ along $\partial \{u > 0\}$, with monotone Weiss energy:
//! $$W(u, r) = r^{-n} \int_{B_r} (|\nabla u|^2 + \Lambda \mathbf{1}_{\{u > 0\}}) \, dx - r^{-(n+1)} \int_{\partial B_r} u^2 \, d\mathcal{H}^{n-1}$$

use crate::error::{Error, Result};

/// Alt-Caffarelli Free-Boundary & Weiss Monotonicity Engine.
#[derive(Debug, Clone, Copy)]
pub struct AltCaffarelliFreeBoundaryWeiss {
    /// Bernoulli phase weight $\Lambda > 0$.
    pub lambda_weight: f64,
    /// Spatial dimension $n \ge 2$.
    pub dim_n: usize,
}

impl Default for AltCaffarelliFreeBoundaryWeiss {
    fn default() -> Self {
        Self {
            lambda_weight: 4.0,
            dim_n: 2,
        }
    }
}

impl AltCaffarelliFreeBoundaryWeiss {
    /// Creates a new Alt-Caffarelli Free-Boundary engine.
    pub fn new(lambda_weight: f64, dim_n: usize) -> Result<Self> {
        if lambda_weight <= 0.0 || dim_n < 2 {
            return Err(Error::Msg(
                "lambda_weight > 0 and dim_n >= 2 required".to_string(),
            ));
        }
        Ok(Self {
            lambda_weight,
            dim_n,
        })
    }

    /// Evaluates the Bernoulli free-boundary gradient slope $\sqrt{\Lambda}$ and the Weiss functional $W(u, r)$ at radius $r > 0$.
    pub fn bernoulli_slope_and_weiss_energy(
        &self,
        radius_r: f64,
        dirichlet_int: f64,
        positive_measure_br: f64,
        boundary_u2_int: f64,
    ) -> Result<(f64, f64)> {
        if radius_r <= 0.0
            || dirichlet_int < 0.0
            || positive_measure_br < 0.0
            || boundary_u2_int < 0.0
        {
            return Err(Error::Msg(
                "radius_r > 0 and non-negative integrals required".to_string(),
            ));
        }
        let n = i32::try_from(self.dim_n)
            .map_err(|_| Error::Msg("dim_n exceeds i32 range".to_string()))?;
        let n1 = n
            .checked_add(1)
            .ok_or_else(|| Error::Msg("dim_n + 1 overflows i32".to_string()))?;
        let slope = self.lambda_weight.sqrt();
        let w_r = (dirichlet_int + self.lambda_weight * positive_measure_br) / radius_r.powi(n)
            - boundary_u2_int / radius_r.powi(n1);
        Ok((slope, w_r))
    }
}
