//! Algorithmic Information Theory, Solomonoff Universal Prior, Levin $Kt$-Complexity & Minimum Description Length.
//!
//! ### 1. Solomonoff Universal Semimeasure Prior $M(x)$ (Solomonoff, 1964)
//! Given a prefix-free universal Turing machine $U$, the **Solomonoff Algorithmic Probability** of an observation $x$
//! sums over all halting programs $p$ that output $x$, weighted by $2^{-|p|}$ (Kraft's inequality $\sum_p 2^{-|p|} \le 1$):
//! $$M(x) = \sum_{p : U(p) = x} 2^{-|p|}$$
//! By the **Coding Theorem (Levin, 1974)**, $-\log_2 M(x) = K(x) + O(1)$, where $K(x)$ is prefix Kolmogorov complexity.
//!
//! ### 2. Levin Universal Search & Time-Bounded $Kt$-Complexity (Levin, 1973)
//! Because Kolmogorov complexity $K(x)$ is uncomputable, **Levin Complexity** penalizes program length $|p|$
//! plus logarithmic runtime $\log_2 t(p)$:
//! $$Kt(p, t) = |p| + \log_2 t$$
//! Levin Universal Search executes candidate programs in increasing order of $Kt(p, t)$, guaranteeing optimal
//! asymptotic search time!

use crate::error::{Error, Result};

/// Candidate hypothesis/program in a Solomonoff-Levin Universal Induction ensemble.
#[derive(Debug, Clone, Copy)]
pub struct AlgorithmicHypothesis {
    /// Prefix-free program description length $|p|$ in bits.
    pub code_length_bits: f64,
    /// Execution step count $t \ge 1$.
    pub runtime_steps: usize,
    /// Empirical negative log-likelihood (data residual bits) $-\log_2 P(D | p)$.
    pub data_misfit_bits: f64,
}

/// Solomonoff Universal Prior, Levin $Kt$-Complexity & MDL Engine.
#[derive(Debug, Clone, Default)]
pub struct SolomonoffLevinEngine {
    /// Registered candidate hypotheses.
    pub hypotheses: Vec<AlgorithmicHypothesis>,
}

impl SolomonoffLevinEngine {
    /// Creates a new Solomonoff-Levin induction engine.
    pub fn new(hypotheses: Vec<AlgorithmicHypothesis>) -> Result<Self> {
        if hypotheses.is_empty() {
            return Err(Error::Msg(
                "At least one hypothesis is required".to_string(),
            ));
        }
        for h in &hypotheses {
            // NaN fails every comparison, so it slipped through and poisoned
            // all downstream aggregates with silent NaNs.
            if h.code_length_bits.is_nan() || h.data_misfit_bits.is_nan() {
                return Err(Error::Msg("Invalid hypothesis parameters".to_string()));
            }
            if h.code_length_bits <= 0.0 || h.runtime_steps == 0 || h.data_misfit_bits < 0.0 {
                return Err(Error::Msg("Invalid hypothesis parameters".to_string()));
            }
        }
        Ok(Self { hypotheses })
    }

    /// Verifies Kraft's Inequality $\sum_i 2^{-|p_i|} \le 1.0$ for prefix-free codes.
    pub fn kraft_inequality_sum(&self) -> f64 {
        self.hypotheses
            .iter()
            .map(|h| 2.0_f64.powf(-h.code_length_bits))
            .sum()
    }

    /// Computes Levin's time-bounded complexity $Kt(p_i) = |p_i| + \log_2(t_i)$ for each hypothesis.
    pub fn levin_kt_complexities(&self) -> Vec<f64> {
        self.hypotheses
            .iter()
            .map(|h| h.code_length_bits + (h.runtime_steps as f64).log2())
            .collect()
    }

    /// Computes normalized Solomonoff posterior weights $w_i \propto 2^{-(|p_i| + \text{misfit}_i)}$ (Minimum Description Length).
    pub fn solomonoff_posterior_weights(&self) -> Vec<f64> {
        let mdl_bits: Vec<f64> = self
            .hypotheses
            .iter()
            .map(|h| h.code_length_bits + h.data_misfit_bits)
            .collect();
        let min_bits = mdl_bits.iter().copied().fold(f64::INFINITY, f64::min);
        let unnorm: Vec<f64> = mdl_bits
            .iter()
            .map(|&b| 2.0_f64.powf(-(b - min_bits)))
            .collect();
        let z: f64 = unnorm.iter().sum();
        unnorm.into_iter().map(|u| u / z).collect()
    }
}
