//! Solomonoff Algorithmic Universal Prior $M(x)$ & Levin Time-Bounded $Kt$-Complexity (Solomonoff, 1964; Levin, 1973).
//!
//! ### 1. Levin $Kt$-Complexity & Universal Search Prior
//! Given a candidate program of description length $|p|$ (in bits) that outputs $x$ in $t(p)$ execution steps,
//! Levin's computable complexity is:
//! $$Kt(x) = \min_{p : U(p) = x} \bigl(|p| + \log_2 t(p)\bigr), \qquad m_t(x) = \sum_{p : U(p) = x} \frac{2^{-|p|}}{t(p)}$$

use crate::error::{Error, Result};

/// Solomonoff Universal Algorithmic Prior & Levin $Kt$-Complexity Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SolomonoffLevinUniversalPrior;

impl SolomonoffLevinUniversalPrior {
    /// Creates a new Solomonoff-Levin Universal Prior engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates Levin's minimum $Kt(x)$ complexity and time-weighted Solomonoff semimeasure $\sum 2^{-|p|} / t(p)$ across candidate programs $(|p|_i, t_i)$.
    pub fn levin_kt_and_universal_prior(
        program_lengths_bits: &[usize],
        runtimes_steps: &[u64],
    ) -> Result<(f64, f64)> {
        let n = program_lengths_bits.len();
        if n == 0 || runtimes_steps.len() != n || runtimes_steps.contains(&0) {
            return Err(Error::Msg(
                "Non-empty matching slices with runtimes >= 1 required".to_string(),
            ));
        }
        let mut min_kt = f64::INFINITY;
        let mut prior_sum = 0.0;
        for i in 0..n {
            let len_b = program_lengths_bits[i] as f64;
            let t = runtimes_steps[i] as f64;
            let kt = len_b + t.log2();
            if kt < min_kt {
                min_kt = kt;
            }
            prior_sum += 2.0_f64.powf(-len_b) / t;
        }
        Ok((min_kt, prior_sum))
    }
}
