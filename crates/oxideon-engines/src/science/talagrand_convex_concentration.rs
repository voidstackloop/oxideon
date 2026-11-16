//! Talagrand Convex Distance Concentration Inequality on Product Spaces (Talagrand, 1995).
//!
//! ### 1. Talagrand Convex Distance $d_T(x, A)$ & Gaussian Tail Bound
//! For a product probability space $\Omega = \prod_{i=1}^n \Omega_i$ and measurable $A \subset \Omega$ with $\mathbb{P}(A) \ge 1/2$:
//! $$\mathbb{P}\!\left(d_T(X, A) \ge t\right) \le 2 \exp\!\left(-\frac{t^2}{4}\right)$$
//! For a $L$-Lipschitz convex function $f$ with median $M$, $\mathbb{P}(|f(X) - M| \ge t) \le 4 \exp(-t^2 / (4 L^2))$.

use crate::error::{Error, Result};

/// Talagrand Convex Distance Concentration Inequality Engine.
#[derive(Debug, Clone, Copy)]
pub struct TalagrandConvexConcentration {
    /// Lipschitz constant $L > 0$ with respect to Euclidean/convex distance.
    pub lipschitz_l: f64,
}

impl Default for TalagrandConvexConcentration {
    fn default() -> Self {
        Self { lipschitz_l: 1.0 }
    }
}

impl TalagrandConvexConcentration {
    /// Creates a new Talagrand Convex Concentration engine.
    pub fn new(lipschitz_l: f64) -> Result<Self> {
        if lipschitz_l <= 0.0 || !lipschitz_l.is_finite() {
            return Err(Error::Msg("lipschitz_l must be > 0".to_string()));
        }
        Ok(Self { lipschitz_l })
    }

    /// Evaluates the Talagrand convex-set tail bound $2 \exp(-t^2 / (4 L^2))$ and two-sided deviation bound from the median.
    pub fn convex_tail_and_median_deviation_bound(&self, deviation_t: f64) -> Result<(f64, f64)> {
        if deviation_t < 0.0 || !deviation_t.is_finite() {
            return Err(Error::Msg("deviation_t must be >= 0".to_string()));
        }
        let exponent = -(deviation_t * deviation_t) / (4.0 * self.lipschitz_l * self.lipschitz_l);
        let one_sided = (2.0 * exponent.exp()).min(1.0);
        let two_sided = (4.0 * exponent.exp()).min(1.0);
        Ok((one_sided, two_sided))
    }
}
