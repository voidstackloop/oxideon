//! Calderón-Zygmund Singular Integral Operators, Pichorides Optimal $L^p$ Hilbert Transform Norm & Riesz Multipliers (Calderón & Zygmund, 1952; Pichorides, 1972).
//!
//! ### 1. Exact $L^p(\mathbb{R})$ Operator Norm of the Hilbert Transform $H$
//! For $1 < p < \infty$, Pichorides proved that the exact $L^p \to L^p$ operator norm of the Hilbert transform is:
//! $$\|H\|_{L^p \to L^p} = \begin{cases} \tan\!\left(\frac{\pi}{2 p}\right) & 1 < p \le 2 \\ \cot\!\left(\frac{\pi}{2 p}\right) & 2 \le p < \infty \end{cases}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Calderón-Zygmund Singular Integral & Pichorides Hilbert Transform Engine.
#[derive(Debug, Clone, Copy)]
pub struct CalderonZygmundSingularIntegral {
    /// Lebesgue exponent $p \in (1, \infty)$.
    pub exponent_p: f64,
}

impl Default for CalderonZygmundSingularIntegral {
    fn default() -> Self {
        Self { exponent_p: 2.0 }
    }
}

impl CalderonZygmundSingularIntegral {
    /// Creates a new Calderón-Zygmund Singular Integral engine.
    pub fn new(exponent_p: f64) -> Result<Self> {
        if exponent_p <= 1.0 || !exponent_p.is_finite() {
            return Err(Error::Msg(
                "Lebesgue exponent p must be in (1, inf)".to_string(),
            ));
        }
        Ok(Self { exponent_p })
    }

    /// Evaluates the exact Pichorides $L^p$ operator norm $\|H\|_{L^p \to L^p}$ and dual exponent $p' = p / (p - 1)$.
    pub fn hilbert_transform_lp_norm_and_dual(&self) -> (f64, f64) {
        let p_dual = self.exponent_p / (self.exponent_p - 1.0);
        let angle = PI / (2.0 * self.exponent_p);
        let norm = if self.exponent_p <= 2.0 {
            angle.tan()
        } else {
            1.0 / angle.tan()
        };
        (norm, p_dual)
    }
}
