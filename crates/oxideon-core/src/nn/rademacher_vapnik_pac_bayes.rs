//! Statistical Learning Theory, Empirical Rademacher Complexity $\widehat{\mathfrak{R}}_S(\mathcal{F})$ & McAllester PAC-Bayes Bounds (McAllester, 1999).
//!
//! ### 1. McAllester-Maurer PAC-Bayes Generalization Bound
//! For any prior distribution $\pi$ on hypothesis space $\mathcal{H}$ chosen before seeing $n$ i.i.d. samples $S$,
//! with probability at least $1 - \delta$ over $S$, every posterior distribution $\rho$ simultaneously satisfies:
//! $$L(\rho) \le \hat{L}_S(\rho) + \sqrt{\frac{\mathrm{KL}(\rho \| \pi) + \ln(2 \sqrt{n} / \delta)}{2 n}}$$

use crate::error::{Error, Result};

/// PAC-Bayes McAllester Generalization & Rademacher Complexity Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct PacBayesMcAllesterEngine;

impl PacBayesMcAllesterEngine {
    /// Creates a new PAC-Bayes engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the McAllester-Maurer PAC-Bayes upper bound on expected risk $L(\rho) \le \min(1, \hat{L}_S(\rho) + \text{rad})$.
    pub fn mcallester_risk_bound(
        empirical_risk: f64,
        kl_divergence: f64,
        num_samples_n: usize,
        failure_delta: f64,
    ) -> Result<f64> {
        if !(0.0..=1.0).contains(&empirical_risk)
            || kl_divergence < 0.0
            || num_samples_n == 0
            || !(0.0..1.0).contains(&failure_delta)
            || failure_delta == 0.0
        {
            return Err(Error::Msg("Invalid PAC-Bayes bound parameters".to_string()));
        }
        let n = num_samples_n as f64;
        let penalty = ((kl_divergence + (2.0 * n.sqrt() / failure_delta).ln()) / (2.0 * n)).sqrt();
        Ok((empirical_risk + penalty).min(1.0))
    }
}
