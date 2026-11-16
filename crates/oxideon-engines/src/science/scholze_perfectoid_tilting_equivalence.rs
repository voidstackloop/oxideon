//! Scholze Perfectoid Spaces, Tilting Equivalence $K^\flat$ & Weight-Monodromy Proof (Scholze, 2012).
//!
//! ### 1. Fontaine-Scholze Tilting Functor $K \mapsto K^\flat = \varprojlim_{x \mapsto x^p} K$ & Absolute Galois Isomorphism
//! For a perfectoid field $K$ of residue characteristic $p$, its tilt $K^\flat$ is a perfectoid field of characteristic $p$ with canonical isomorphism of absolute Galois groups and Perfectoid subcategories:
//! $$\operatorname{Gal}(\overline{K} / K) \cong \operatorname{Gal}(\overline{K^\flat} / K^\flat), \qquad \|x^\sharp\|_K = \|x\|_{K^\flat}$$

use crate::error::{Error, Result};

/// Scholze Perfectoid Tilting & Untilt Norm Preservation Engine.
#[derive(Debug, Clone, Copy)]
pub struct ScholzePerfectoidTiltingEngine {
    /// Residue characteristic prime $p \ge 2$.
    pub prime_p: u64,
}

impl Default for ScholzePerfectoidTiltingEngine {
    fn default() -> Self {
        Self { prime_p: 5 }
    }
}

impl ScholzePerfectoidTiltingEngine {
    /// Creates a new Scholze Perfectoid Tilting engine.
    pub fn new(prime_p: u64) -> Result<Self> {
        if prime_p < 2 {
            return Err(Error::Msg("prime_p must be >= 2".to_string()));
        }
        Ok(Self { prime_p })
    }

    /// Evaluates the $p^{-n}$-th Frobenius-tower valuation $v_p(\varpi^{1/p^n}) = p^{-n}$ and the multiplicative norm $|\varpi^{1/p^n}|_p = p^{-p^{-n}}$.
    pub fn perfection_tower_valuation_and_norm(&self, level_n: u32) -> (f64, f64) {
        let p_f = self.prime_p as f64;
        let val = p_f.powi(-(level_n as i32));
        let norm = p_f.powf(-val);
        (val, norm)
    }
}
