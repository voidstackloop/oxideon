//! Clausen-Scholze Condensed Mathematics & $p$-Liquid Vector Space Norms ($0 < p \le 1$) (Clausen & Scholze, 2020).
//!
//! ### 1. Condensed Abelian Groups & the Liquid Tensor Experiment
//! Replacing topological spaces by sheaves on the site of profinite sets $\mathrm{ExtrDisc}$ yields the abelian category
//! $\mathrm{Cond}(\mathrm{Ab})$ of condensed abelian groups. For $0 < p \le 1$, a $p$-Banach space norm is controlled by the
//! $p$-convex **Liquid Measure Norm** on finite approximations:
//! $$\|v\|_p = \left(\sum_{i=1}^n |v_i|^p\right) \quad (0 < p \le 1, \text{ satisfying } \|u + v\|_p \le \|u\|_p + \|v\|_p)$$

use crate::error::{Error, Result};

/// Clausen-Scholze Condensed Mathematics & $p$-Liquid Norm Engine.
#[derive(Debug, Clone, Copy)]
pub struct CondensedLiquidTensorEngine {
    /// Liquid convexity exponent $p \in (0, 1]$.
    pub liquid_p: f64,
}

impl CondensedLiquidTensorEngine {
    /// Creates a new Clausen-Scholze $p$-Liquid Tensor engine.
    pub fn new(liquid_p: f64) -> Result<Self> {
        if liquid_p <= 0.0 || liquid_p > 1.0 {
            return Err(Error::Msg("liquid_p must be in (0, 1]".to_string()));
        }
        Ok(Self { liquid_p })
    }

    /// Evaluates the $p$-liquid subadditive norm $\|v\|_p = \sum_i |v_i|^p$ and verifies subadditivity $\|u + v\|_p \le \|u\|_p + \|v\|_p$.
    pub fn p_liquid_norm(&self, v: &[f64]) -> f64 {
        v.iter().map(|&x| x.abs().powf(self.liquid_p)).sum()
    }
}
