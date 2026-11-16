//! Lurie Quasi-Categories ($(\infty, 1)$-Categories), Inner Horn Lifting $\Lambda^n_k \to \Delta^n$ & Straightening-Unstraightening (Joyal, 2002; Lurie, 2009).
//!
//! ### 1. Simplicial $n$-Simplex Faces $\binom{n+1}{k+1}$ & Inner Horn Count $n - 1$
//! A simplicial set $X$ is an $\infty$-category (quasi-category) iff every inner horn $\Lambda^n_k \to X$ for $0 < k < n$ admits a filler $\Delta^n \to X$; an $n$-simplex $\Delta^n$ has $n + 1$ codimension-1 faces, of which $n - 1$ correspond to inner horns.

use crate::error::{Error, Result};

/// Lurie $(\infty, 1)$-Category & Inner Horn Combinatorics Engine.
#[derive(Debug, Clone, Copy)]
pub struct LurieInfinityCategoryEngine {
    /// Simplex dimension $n \ge 2$.
    pub simplex_dim_n: usize,
}

impl Default for LurieInfinityCategoryEngine {
    fn default() -> Self {
        Self { simplex_dim_n: 3 }
    }
}

impl LurieInfinityCategoryEngine {
    /// Creates a new Lurie Infinity-Category engine.
    pub fn new(simplex_dim_n: usize) -> Result<Self> {
        if !(2..=30).contains(&simplex_dim_n) {
            return Err(Error::Msg("simplex_dim_n must be in 2..=30".to_string()));
        }
        Ok(Self { simplex_dim_n })
    }

    /// Evaluates the number of inner horns $n - 1$ in dimension $n$ and the total number of non-degenerate sub-simplices $2^{n+1} - 1$ of $\Delta^n$.
    pub fn inner_horn_count_and_subsimplices(&self) -> (usize, u64) {
        let inner = self.simplex_dim_n - 1;
        let sub = (1_u64 << (self.simplex_dim_n + 1)) - 1;
        (inner, sub)
    }
}
