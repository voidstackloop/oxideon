//! Tutte Polynomial $T_G(x, y)$, Chromatic/Flow Specialization & Deletion-Contraction Recurrence (Tutte, 1954).
//!
//! ### 1. Universal Deletion-Contraction Invariant
//! For a connected graph $G = (V, E)$, the Tutte polynomial $T_G(x, y)$ specializes to the number of spanning trees $T_G(1, 1)$,
//! forest count $T_G(2, 1)$, acyclic orientations $T_G(2, 0)$, and chromatic polynomial $P_G(k) = (-1)^{|V|-1} k \, T_G(1 - k, 0)$.

use crate::error::{Error, Result};

/// Tutte Polynomial, Chromatic Specialization & Matroid Rank Engine.
#[derive(Debug, Clone, Copy)]
pub struct TuttePolynomialMatroidEngine {
    /// Cycle length $n \ge 3$ for the cycle graph $C_n$ (whose Tutte polynomial is $y + x + x^2 + \dots + x^{n-1}$).
    pub cycle_n: usize,
}

impl Default for TuttePolynomialMatroidEngine {
    fn default() -> Self {
        Self { cycle_n: 4 }
    }
}

impl TuttePolynomialMatroidEngine {
    /// Creates a new Tutte Polynomial Matroid engine for $C_n$.
    pub fn new(cycle_n: usize) -> Result<Self> {
        if cycle_n < 3 {
            return Err(Error::Msg("cycle_n must be >= 3".to_string()));
        }
        Ok(Self { cycle_n })
    }

    /// Evaluates $T_{C_n}(x, y) = y + \sum_{k=1}^{n-1} x^k$, spanning tree count $T_{C_n}(1, 1) = n$, and acyclic orientations $T_{C_n}(2, 0) = 2^n - 2$.
    pub fn evaluate_tutte_and_invariants(&self, x: f64, y: f64) -> (f64, usize, u64) {
        let mut t_val = y;
        let mut x_pow = 1.0;
        for _ in 1..self.cycle_n {
            x_pow *= x;
            t_val += x_pow;
        }
        let spanning_trees = self.cycle_n;
        let acyclic_orientations = (1u64 << self.cycle_n) - 2;
        (t_val, spanning_trees, acyclic_orientations)
    }
}
