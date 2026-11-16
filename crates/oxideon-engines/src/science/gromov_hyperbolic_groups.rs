//! Gromov $\delta$-Hyperbolic Metric Spaces, Gromov Product & Four-Point Condition (Gromov, 1987).
//!
//! ### 1. Gromov Product & $\delta$-Hyperbolicity
//! Given a basepoint $w \in X$, the Gromov product is $(x | y)_w = \frac{1}{2}(d(x, w) + d(y, w) - d(x, y))$.
//! A geodesic metric space $X$ is $\delta$-hyperbolic if for all $x, y, z, w \in X$:
//! $$(x | z)_w \ge \min\{(x | y)_w, (y | z)_w\} - \delta$$

use crate::error::{Error, Result};

/// Gromov $\delta$-Hyperbolic Metric Space & Morse Geodesic Stability Engine.
#[derive(Debug, Clone, Copy)]
pub struct GromovDeltaHyperbolicEngine {
    /// Hyperbolicity constant $\delta \ge 0$.
    pub delta: f64,
}

impl Default for GromovDeltaHyperbolicEngine {
    fn default() -> Self {
        Self { delta: 0.0 }
    }
}

impl GromovDeltaHyperbolicEngine {
    /// Creates a new Gromov $\delta$-Hyperbolic engine.
    pub fn new(delta: f64) -> Result<Self> {
        if delta < 0.0 || !delta.is_finite() {
            return Err(Error::Msg(
                "Gromov hyperbolicity delta must be >= 0".to_string(),
            ));
        }
        Ok(Self { delta })
    }

    /// Evaluates the Gromov product $(x | y)_w = \frac{1}{2}(d_{xw} + d_{yw} - d_{xy})$.
    pub fn gromov_product(d_xw: f64, d_yw: f64, d_xy: f64) -> f64 {
        0.5 * (d_xw + d_yw - d_xy)
    }

    /// Verifies the four-point $\delta$-hyperbolicity defect $\max(d_{xy}+d_{zw}, d_{xz}+d_{yw}, d_{xw}+d_{yz}) - \text{mid} \le 2\delta$.
    pub fn four_point_defect_and_check(
        &self,
        d_xy: f64,
        d_zw: f64,
        d_xz: f64,
        d_yw: f64,
        d_xw: f64,
        d_yz: f64,
    ) -> (f64, bool) {
        let mut s = [d_xy + d_zw, d_xz + d_yw, d_xw + d_yz];
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let defect = 0.5 * (s[2] - s[1]);
        (defect, defect <= self.delta + 1e-12)
    }
}
