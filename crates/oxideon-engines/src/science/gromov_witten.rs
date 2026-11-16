//! Gromov-Witten Invariants, Quantum Cohomology Ring $QH^*(X)$, WDVV Equations & Kontsevich Rational Curve Counts.
//!
//! ### 1. Kontsevich's Recursion for Rational Curves on $\mathbb{P}^2$ (Kontsevich, 1994)
//! Let $N_d$ denote the Gromov-Witten invariant counting rational degree-$d$ curves in $\mathbb{CP}^2$ passing through
//! $3d - 1$ general points. By the **Witten-Dijkgraaf-Verlinde-Verlinde (WDVV)** associativity equation of
//! the Gromov-Witten prepotential $\mathcal{F}_{\text{GW}}$, $N_d$ satisfies Kontsevich's exact recursion ($N_1 = 1$):
//! $$N_d = \sum_{d_1 + d_2 = d} N_{d_1} N_{d_2} \left( d_1^2 d_2^2 \binom{3d - 4}{3d_1 - 2} - d_1^3 d_2 \binom{3d - 4}{3d_1 - 1} \right)$$
//! Yielding the famous enumerative sequence: $N_1 = 1, \; N_2 = 1, \; N_3 = 12, \; N_4 = 620, \; N_5 = 87304, \dots$

use crate::error::{Error, Result};

/// Gromov-Witten Quantum Cohomology & Kontsevich WDVV Recursion Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct GromovWittenWdvvEngine;

impl GromovWittenWdvvEngine {
    /// Creates a new Gromov-Witten WDVV engine.
    pub fn new() -> Self {
        Self
    }

    fn binom(n: u64, k: u64) -> u64 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut res = 1_u64;
        for i in 0..k {
            res = res * (n - i) / (i + 1);
        }
        res
    }

    /// Computes Kontsevich's exact Gromov-Witten rational curve count $N_d$ on $\mathbb{CP}^2$ via WDVV recursion.
    pub fn kontsevich_rational_curve_count(degree_d: usize) -> Result<u64> {
        if !(1..=8).contains(&degree_d) {
            return Err(Error::Msg("degree_d must be in 1..=8".to_string()));
        }
        let mut n_d = vec![0_u64; degree_d + 1];
        n_d[1] = 1;
        for d in 2..=degree_d {
            let mut sum = 0_i128;
            let top = (3 * d - 4) as u64;
            for d1 in 1..d {
                let d2 = d - d1;
                let d1_i = d1 as i128;
                let d2_i = d2 as i128;
                let c1 = Self::binom(top, (3 * d1 - 2) as u64) as i128;
                let c2 = Self::binom(top, (3 * d1 - 1) as u64) as i128;
                let term = (n_d[d1] as i128)
                    * (n_d[d2] as i128)
                    * (d1_i * d1_i * d2_i * d2_i * c1 - d1_i.pow(3) * d2_i * c2);
                sum += term;
            }
            n_d[d] = sum as u64;
        }
        Ok(n_d[degree_d])
    }
}
