//! Clausen-Scholze Condensed Mathematics & Liquid Tensor Experiment $p$-Liquid Norm $\|\cdot\|_p$ (Clausen & Scholze, 2019/2022).
//!
//! ### 1. $p$-Liquid Measure Space $\mathcal{M}_{< p}(S)$ & Ext-Vanishing for $0 < p' < p \le 1$
//! For a profinite set $S$ and $0 < p \le 1$, the $p$-liquid norm of a finitely supported real sequence $(c_i)$ is $\|(c_i)\|_p = (\sum_i |c_i|^p)^{1/p}$, and the Liquid Tensor Theorem establishes $\operatorname{Ext}^i_{\mathrm{Cond}(\mathbb{R})}(\mathcal{M}_{< p'}(S), V) = 0$ for all $i \ge 1$ and $p$-Banach spaces $V$ whenever $0 < p' < p \le 1$.

use crate::error::{Error, Result};

/// Clausen-Scholze Condensed Mathematics & $p$-Liquid Tensor Norm Engine.
#[derive(Debug, Clone, Copy)]
pub struct ClausenScholzeLiquidTensor {
    /// Liquid exponent $p \in (0, 1]$.
    pub liquid_exponent_p: f64,
}

impl Default for ClausenScholzeLiquidTensor {
    fn default() -> Self {
        Self {
            liquid_exponent_p: 0.5,
        }
    }
}

impl ClausenScholzeLiquidTensor {
    /// Creates a new Clausen-Scholze Liquid Tensor engine.
    pub fn new(liquid_exponent_p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&liquid_exponent_p) || liquid_exponent_p == 0.0 {
            return Err(Error::Msg(
                "liquid_exponent_p must be in (0, 1]".to_string(),
            ));
        }
        Ok(Self { liquid_exponent_p })
    }

    /// Evaluates the $p$-quasi-Banach norm $(\sum_i |c_i|^p)^{1/p}$ and checks the Clausen-Scholze Ext-vanishing condition $p' < p$.
    pub fn p_liquid_norm_and_ext_vanishing(
        &self,
        coeffs: &[f64],
        test_p_prime: f64,
    ) -> Result<(f64, bool)> {
        if coeffs.is_empty() || test_p_prime <= 0.0 {
            return Err(Error::Msg(
                "coeffs must be non-empty and test_p_prime > 0".to_string(),
            ));
        }
        let sum_p: f64 = coeffs
            .iter()
            .map(|&c| c.abs().powf(self.liquid_exponent_p))
            .sum();
        let quasi_norm = sum_p.powf(1.0 / self.liquid_exponent_p);
        Ok((quasi_norm, test_p_prime < self.liquid_exponent_p))
    }
}
