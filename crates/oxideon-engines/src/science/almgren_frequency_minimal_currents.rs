//! Almgren Big Regularity Theorem for Area-Minimizing Currents & Almgren Frequency Function $N(x_0, r)$ (Almgren, 1983/2000; De Lellis & Spadaro, 2016).
//!
//! ### 1. Almgren Monotone Frequency Function $N(r)$ & Codimension-2 Singular Set Bound
//! For a $Q$-valued Dirichlet-minimizing map $u$ on $B_r(x_0) \subset \mathbb{R}^m$, Almgren's frequency function is:
//! $$N(r) = \frac{r \int_{B_r} |D u|^2 \, dx}{\int_{\partial B_r} |u|^2 \, d\mathcal{H}^{m-1}}$$
//! By Almgren's monotonicity theorem, $r \mapsto N(r)$ is non-decreasing, and the singular set of an area-minimizing $m$-current has Hausdorff dimension $\le m - 2$.

use crate::error::{Error, Result};

/// Almgren Frequency Function & Area-Minimizing Current Regularity Engine.
#[derive(Debug, Clone, Copy)]
pub struct AlmgrenFrequencyCurrentsEngine {
    /// Current dimension $m \ge 2$.
    pub current_dim_m: usize,
}

impl Default for AlmgrenFrequencyCurrentsEngine {
    fn default() -> Self {
        Self { current_dim_m: 4 }
    }
}

impl AlmgrenFrequencyCurrentsEngine {
    /// Creates a new Almgren Frequency & Minimal Currents engine.
    pub fn new(current_dim_m: usize) -> Result<Self> {
        if current_dim_m < 2 {
            return Err(Error::Msg("current_dim_m must be >= 2".to_string()));
        }
        Ok(Self { current_dim_m })
    }

    /// Evaluates Almgren's frequency $N(r) = r D(r) / H(r)$ and the maximum singular set Hausdorff dimension $m - 2$ in higher codimension.
    pub fn frequency_and_max_singular_dim(
        &self,
        radius_r: f64,
        dirichlet_energy_dr: f64,
        boundary_l2_hr: f64,
    ) -> Result<(f64, usize)> {
        if radius_r <= 0.0 || dirichlet_energy_dr < 0.0 || boundary_l2_hr <= 0.0 {
            return Err(Error::Msg(
                "radius_r > 0, dirichlet_energy_dr >= 0, boundary_l2_hr > 0 required".to_string(),
            ));
        }
        let n_r = radius_r * dirichlet_energy_dr / boundary_l2_hr;
        Ok((n_r, self.current_dim_m - 2))
    }
}
