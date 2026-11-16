//! Drinfeld-Lafforgue Shtukas over Function Fields $\mathbb{F}_q(X)$ & Global Langlands Correspondence (Drinfeld, 1987; Lafforgue, 2002).
//!
//! ### 1. Dimension of Moduli Stack of Rank-$r$ Shtukas with $n$ Legs
//! Over a smooth projective curve $X / \mathbb{F}_q$, the moduli stack $\mathrm{Sht}_G^{r, n}$ of rank-$r$ shtukas with $n$ legs
//! has relative dimension $(r - 1) n$ over $X^n$ (or $n \dim(G/B)$ for a general split reductive group $G$).

use crate::error::{Error, Result};

/// Drinfeld-Lafforgue Shtuka Moduli & Function-Field Langlands Engine.
#[derive(Debug, Clone, Copy)]
pub struct DrinfeldShtukaLanglandsEngine {
    /// Rank $r \ge 2$ of the vector bundle / $GL_r$-shtuka.
    pub rank_r: usize,
    /// Base field cardinality $q \ge 2$.
    pub field_q: u64,
}

impl Default for DrinfeldShtukaLanglandsEngine {
    fn default() -> Self {
        Self {
            rank_r: 2,
            field_q: 3,
        }
    }
}

impl DrinfeldShtukaLanglandsEngine {
    /// Creates a new Drinfeld-Lafforgue Shtuka engine.
    pub fn new(rank_r: usize, field_q: u64) -> Result<Self> {
        if rank_r < 2 || field_q < 2 {
            return Err(Error::Msg(
                "rank_r >= 2 and field_q >= 2 required".to_string(),
            ));
        }
        Ok(Self { rank_r, field_q })
    }

    /// Computes the total dimension $d = r \cdot n$ of the moduli stack of $GL_r$ shtukas with $n$ legs (with trivialized determinant $d = (r - 1) n + n$).
    pub fn shtuka_moduli_dimension(&self, num_legs_n: usize) -> (usize, usize) {
        let flag_dim = self.rank_r * (self.rank_r - 1) / 2;
        let pgl_dim = (self.rank_r - 1) * num_legs_n;
        (pgl_dim, flag_dim * num_legs_n)
    }
}
