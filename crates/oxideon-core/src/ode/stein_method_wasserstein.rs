//! Stein's Method for Normal & Poisson Approximation & Nualart-Peccati Fourth-Moment Theorem (Stein, 1972; Nualart & Peccati, 2005).
//!
//! ### 1. Nualart-Peccati Fourth-Moment Theorem on Wiener Chaos
//! For a sequence of normalized random variables $F_n$ in the $q$-th Wiener chaos ($\mathbb{E}\[F_n\] = 0, \mathbb{E}\[F_n^2\] = 1$),
//! convergence in total variation and Wasserstein-1 distance to $Z \sim \mathcal{N}(0, 1)$ is controlled solely by the
//! fourth cumulant $\kappa_4(F_n) = \mathbb{E}[F_n^4] - 3 \ge 0$:
//! $$d_{\mathrm{TV}}(F_n, Z) \le 2 \sqrt{\frac{q - 1}{3 q} \left|\mathbb{E}[F_n^4] - 3\right|}$$

use crate::error::{Error, Result};

/// Stein's Method & Nualart-Peccati Wiener Chaos Fourth-Moment Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SteinPoissonCharacterization;

impl SteinPoissonCharacterization {
    /// Creates a new Stein's Method & Fourth-Moment engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the Nualart-Peccati-Nourdin Total Variation bound $2 \sqrt{\frac{q-1}{3q} |\mathbb{E}[F^4] - 3|}$ in the $q$-th Wiener chaos ($q \ge 2$).
    pub fn fourth_moment_tv_bound(chaos_order_q: usize, fourth_moment: f64) -> Result<f64> {
        if chaos_order_q < 2 || fourth_moment < 1.0 {
            return Err(Error::Msg(
                "chaos_order_q >= 2 and fourth_moment >= 1.0 required".to_string(),
            ));
        }
        let q = chaos_order_q as f64;
        let kappa4 = (fourth_moment - 3.0).abs();
        Ok(2.0 * ((q - 1.0) / (3.0 * q) * kappa4).sqrt())
    }

    /// Computes the Barbour-Hall Stein-Chen Poisson total variation bound $\lambda^{-1} (1 - e^{-\lambda}) \sum_{i=1}^n p_i^2$ for independent Bernoulli sum.
    pub fn stein_chen_poisson_bound(bernoulli_probs: &[f64]) -> Result<f64> {
        if bernoulli_probs.is_empty() || bernoulli_probs.iter().any(|&p| !(0.0..=1.0).contains(&p))
        {
            return Err(Error::Msg(
                "Valid Bernoulli probabilities in [0, 1] required".to_string(),
            ));
        }
        let lam: f64 = bernoulli_probs.iter().sum();
        if lam <= 0.0 {
            return Ok(0.0);
        }
        let sum_sq: f64 = bernoulli_probs.iter().map(|&p| p * p).sum();
        Ok((1.0 - (-lam).exp()) / lam * sum_sq)
    }
}
