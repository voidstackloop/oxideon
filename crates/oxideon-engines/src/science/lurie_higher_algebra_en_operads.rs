//! Lurie Higher Algebra, $\mathbb{E}_n$-Little Cubes Operads, Dunn Additivity & Factorization Homology $\int_M A$ (Lurie, 2017).
//!
//! ### 1. Dunn Additivity $\mathbb{E}_m \otimes \mathbb{E}_n \simeq \mathbb{E}_{m+n}$ & Poisson Bracket Degree
//! In Lurie's $\infty$-categorical *Higher Algebra*, the homology $H_*(\mathbb{E}_n)$ of the little $n$-disks operad ($n \ge 2$)
//! is the $(n-1)$-shifted **Gerstenhaber / Poisson Operad** $\mathrm{Pois}_n$ with Lie bracket of homological degree $n - 1$,
//! and Dunn's Additivity Theorem states $\mathrm{Alg}_{\mathbb{E}_m}(\mathrm{Alg}_{\mathbb{E}_n}(\mathcal{C})) \simeq \mathrm{Alg}_{\mathbb{E}_{m+n}}(\mathcal{C})$.

use crate::error::{Error, Result};

/// Lurie $\mathbb{E}_n$-Operad Dunn Additivity & Factorization Homology Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct LurieLittleDisksFactorization;

impl LurieLittleDisksFactorization {
    /// Creates a new Lurie $\mathbb{E}_n$-Factorization Homology engine.
    pub fn new() -> Self {
        Self
    }

    /// Applies Dunn Additivity $\mathbb{E}_m \otimes \mathbb{E}_n \simeq \mathbb{E}_{m+n}$ and returns `(total_en_level, browder_bracket_degree)` where degree $= m + n - 1$.
    pub fn dunn_additivity_and_browder_degree(m: usize, n: usize) -> Result<(usize, usize)> {
        if m + n == 0 {
            return Err(Error::Msg("m + n must be >= 1".to_string()));
        }
        let total = m + n;
        Ok((total, total - 1))
    }
}
