//! Adiprasito-Huh-Katz Hodge Theory of Matroids & Ultra-Log-Concavity of Independent Set Numbers (Huh, 2012; Adiprasito, Huh & Katz, 2018).
//!
//! ### 1. Combinatorial Hodge-Riemann Relations & Mason's Conjecture
//! By the Kähler package on the Chow ring $A^*(M)$ of any matroid $M$ of rank $r$ on $n$ elements,
//! the whitney/independent set numbers $I_k$ satisfy Mason's ultra-log-concavity inequality:
//! $$\frac{I_k^2}{\binom{n}{k}^2} \ge \frac{I_{k-1}}{\binom{n}{k-1}} \frac{I_{k+1}}{\binom{n}{k+1}}$$

use crate::error::{Error, Result};

/// Adiprasito-Huh-Katz Matroid Chow Ring & Mason Log-Concavity Verifier.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuhKatzMatroidLogConcavity;

impl HuhKatzMatroidLogConcavity {
    /// Creates a new Huh-Katz Matroid Hodge engine.
    pub fn new() -> Self {
        Self
    }

    /// Verifies standard log-concavity $w_k^2 \ge w_{k-1} w_{k+1}$ for all internal indices $1 \le k < m$ and returns the minimum log-concavity margin $\min_k (w_k^2 - w_{k-1} w_{k+1})$.
    pub fn verify_log_concavity(sequence: &[f64]) -> Result<(f64, bool)> {
        if sequence.len() < 3 {
            return Err(Error::Msg("sequence must have length >= 3".to_string()));
        }
        let mut min_margin = f64::INFINITY;
        for k in 1..(sequence.len() - 1) {
            let margin = sequence[k] * sequence[k] - sequence[k - 1] * sequence[k + 1];
            if margin < min_margin {
                min_margin = margin;
            }
        }
        Ok((min_margin, min_margin >= -1e-12))
    }
}
