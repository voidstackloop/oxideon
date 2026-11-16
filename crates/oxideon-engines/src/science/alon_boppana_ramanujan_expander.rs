//! Alon-Boppana Spectral Bound & Lubotzky-Phillips-Sarnak Ramanujan Expander Graphs (Alon, 1986; Lubotzky, Phillips & Sarnak, 1988).
//!
//! ### 1. Alon-Boppana Lower Bound $\lambda_2 \ge 2\sqrt{d - 1} - o(1)$ & Expander Mixing Lemma
//! For any $d$-regular graph $G = (V, E)$, the second largest adjacency eigenvalue in magnitude $\lambda(G) = \max_{|\lambda_i| < d} |\lambda_i|$ satisfies:
//! $$\lambda(G) \le 2 \sqrt{d - 1} \quad (\text{Ramanujan}), \qquad \left|e(S, T) - \frac{d |S| |T|}{n}\right| \le \lambda(G) \sqrt{|S| |T|}$$

use crate::error::{Error, Result};

/// Alon-Boppana Spectral Bound & Ramanujan Expander Mixing Engine.
#[derive(Debug, Clone, Copy)]
pub struct AlonBoppanaRamanujanExpander {
    /// Regularity degree $d \ge 3$.
    pub degree_d: usize,
}

impl Default for AlonBoppanaRamanujanExpander {
    fn default() -> Self {
        Self { degree_d: 6 }
    }
}

impl AlonBoppanaRamanujanExpander {
    /// Creates a new Alon-Boppana Ramanujan Expander engine.
    pub fn new(degree_d: usize) -> Result<Self> {
        if degree_d < 3 {
            return Err(Error::Msg("degree_d must be >= 3".to_string()));
        }
        Ok(Self { degree_d })
    }

    /// Evaluates the Ramanujan spectral threshold $2\sqrt{d - 1}$, spectral gap $d - \lambda_2$, and whether $\lambda_2$ satisfies the Ramanujan bound.
    pub fn ramanujan_threshold_and_spectral_gap(&self, lambda2: f64) -> Result<(f64, f64, bool)> {
        if lambda2 < 0.0 || lambda2 > self.degree_d as f64 {
            return Err(Error::Msg("lambda2 must be in [0, degree_d]".to_string()));
        }
        let ramanujan_bound = 2.0 * ((self.degree_d - 1) as f64).sqrt();
        let gap = self.degree_d as f64 - lambda2;
        Ok((ramanujan_bound, gap, lambda2 <= ramanujan_bound + 1e-12))
    }
}
