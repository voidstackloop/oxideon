//! Hamilton-Perelman Ricci Flow $\partial_t g_{ij} = -2 R_{ij}$ & Perelman $\mathcal{W}$-Entropy Monotonicity (Perelman, 2002).
//!
//! ### 1. Shrinking Round Sphere $S^n$ under Hamilton's Ricci Flow
//! On an $n$-dimensional round sphere of initial radius $r_0 > 0$, the Ricci tensor is $R_{ij} = (n - 1) g_{ij} / r^2$,
//! so Hamilton's Ricci flow $\partial_t g = -2 \mathrm{Ric}$ evolves the radius via:
//! $$\frac{d(r^2)}{dt} = -2(n - 1) \quad \Longrightarrow \quad r^2(t) = r_0^2 - 2(n - 1) t$$
//! reaching finite-time extinction at singularity time $T_{\text{ext}} = \frac{r_0^2}{2(n - 1)}$.

use crate::error::{Error, Result};

/// Hamilton-Perelman Ricci Flow & Shrinking Soliton Engine.
#[derive(Debug, Clone, Copy)]
pub struct PerelmanRicciFlowEngine {
    /// Manifold dimension $n \ge 2$.
    pub dim_n: usize,
    /// Initial sphere radius $r_0 > 0$.
    pub initial_radius: f64,
}

impl PerelmanRicciFlowEngine {
    /// Creates a new Hamilton-Perelman Ricci Flow engine.
    pub fn new(dim_n: usize, initial_radius: f64) -> Result<Self> {
        if dim_n < 2 || initial_radius <= 0.0 {
            return Err(Error::Msg(
                "dim_n >= 2 and initial_radius > 0 required".to_string(),
            ));
        }
        Ok(Self {
            dim_n,
            initial_radius,
        })
    }

    /// Computes the exact finite-time extinction singularity time $T_{\text{ext}} = \frac{r_0^2}{2(n - 1)}$.
    pub fn extinction_time(&self) -> f64 {
        (self.initial_radius * self.initial_radius) / (2.0 * ((self.dim_n - 1) as f64))
    }

    /// Computes the evolved radius $r(t) = \sqrt{r_0^2 - 2(n-1)t}$ and scalar curvature $R(t) = \frac{n(n-1)}{r(t)^2}$ at $t < T_{\text{ext}}$.
    pub fn radius_and_scalar_curvature(&self, t: f64) -> Result<(f64, f64)> {
        let r2 = self.initial_radius * self.initial_radius - 2.0 * ((self.dim_n - 1) as f64) * t;
        if r2 <= 0.0 {
            return Err(Error::Msg(
                "Time t exceeds Ricci flow singularity extinction time".to_string(),
            ));
        }
        let scalar_r = ((self.dim_n * (self.dim_n - 1)) as f64) / r2;
        Ok((r2.sqrt(), scalar_r))
    }
}
