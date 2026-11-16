//! Kazhdan Property (T), Alon-Boppana Spectral Bound & Lubotzky-Phillips-Sarnak Ramanujan Graphs (Kazhdan, 1967; LPS, 1988).
//!
//! ### 1. Ramanujan Spectral Threshold
//! A $d$-regular graph $G$ is a Ramanujan expander if every nontrivial adjacency eigenvalue $\lambda$ ($|\lambda| < d$) satisfies:
//! $$|\lambda| \le 2\sqrt{d - 1}$$

use crate::error::{Error, Result};

/// Kazhdan Property (T) & Ramanujan Expander Graph Spectral Engine.
#[derive(Debug, Clone, Copy)]
pub struct KazhdanPropertyTExpander {
    /// Regularity degree $d \ge 3$.
    pub degree_d: usize,
}

impl Default for KazhdanPropertyTExpander {
    fn default() -> Self {
        Self { degree_d: 6 }
    }
}

impl KazhdanPropertyTExpander {
    /// Creates a new Kazhdan Property (T) & Ramanujan Expander engine.
    pub fn new(degree_d: usize) -> Result<Self> {
        if degree_d < 3 {
            return Err(Error::Msg("Regularity degree d must be >= 3".to_string()));
        }
        Ok(Self { degree_d })
    }

    /// Evaluates the Alon-Boppana / Ramanujan bound $2\sqrt{d - 1}$, spectral gap $d - |\lambda_1|$, and checks the Ramanujan condition.
    pub fn ramanujan_bound_and_gap(&self, second_eigenvalue_abs: f64) -> (f64, f64, bool) {
        let d = self.degree_d as f64;
        let bound = 2.0 * (d - 1.0).sqrt();
        let gap = d - second_eigenvalue_abs;
        (bound, gap, second_eigenvalue_abs <= bound + 1e-12)
    }
}
