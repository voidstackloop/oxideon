//! Hubbell Unified Neutral Theory of Biodiversity & Ewens Sampling Formula (Ewens, 1972; Hubbell, 2001).
//!
//! ### 1. Expected Species Richness under Ewens-Hubbell Sampling
//! Given fundamental biodiversity number $\theta > 0$ and metacommunity sample size $J \ge 1$,
//! the expected number of distinct species (or alleles) in the sample is:
//! $$\mathbb{E}\[S_J\] = \sum_{k=0}^{J-1} \frac{\theta}{\theta + k}$$

use crate::error::{Error, Result};

/// Hubbell Neutral Biodiversity & Ewens Sampling Formula Engine.
#[derive(Debug, Clone, Copy)]
pub struct HubbellEwensBiodiversityEngine {
    /// Fundamental biodiversity / mutation parameter $\theta > 0$.
    pub theta: f64,
}

impl Default for HubbellEwensBiodiversityEngine {
    fn default() -> Self {
        Self { theta: 10.0 }
    }
}

impl HubbellEwensBiodiversityEngine {
    /// Creates a new Hubbell-Ewens Biodiversity engine.
    pub fn new(theta: f64) -> Result<Self> {
        if theta <= 0.0 || !theta.is_finite() {
            return Err(Error::Msg(
                "Biodiversity parameter theta must be > 0".to_string(),
            ));
        }
        Ok(Self { theta })
    }

    /// Evaluates the exact Ewens expected species richness $\mathbb{E}\[S_J\]$ and variance $\mathrm{Var}(S_J) = \sum_{k=0}^{J-1} \frac{\theta k}{(\theta + k)^2}$ for sample size $J$.
    pub fn expected_richness_and_variance(&self, sample_size_j: usize) -> Result<(f64, f64)> {
        if sample_size_j == 0 {
            return Err(Error::Msg("sample_size_j must be >= 1".to_string()));
        }
        crate::limits::check_iterations("sample_size_j", sample_size_j)?;
        let mut mean_s = 0.0;
        let mut var_s = 0.0;
        for k in 0..sample_size_j {
            let kf = k as f64;
            let p_k = self.theta / (self.theta + kf);
            mean_s += p_k;
            var_s += p_k * (1.0 - p_k);
        }
        Ok((mean_s, var_s))
    }
}
