//! Quillen Complex Cobordism $MU_* \cong L$, Formal Group Laws of Height $h$ & Morava $K$-Theory $K(h)$ (Quillen, 1969; Ravenel, 1984).
//!
//! ### 1. Chromatic Height Stratification & Periodicity Degree $|v_h| = 2(p^h - 1)$
//! By Quillen's Theorem, the complex cobordism ring $MU_*$ is canonically isomorphic to Lazard's universal ring $L$
//! classifying 1-dimensional commutative formal group laws $F(X, Y)$. At prime $p$ and chromatic height $h \ge 1$,
//! the Morava $K$-theory spectrum $K(h)$ has coefficient ring $K(h)_*(\mathrm{pt}) = \mathbb{F}_p[v_h, v_h^{-1}]$ with grading:
//! $$|v_h| = 2(p^h - 1)$$

use crate::error::{Error, Result};

/// Chromatic Homotopy, Formal Group Height & Morava $K(h)$ Engine.
#[derive(Debug, Clone, Copy)]
pub struct ChromaticMoravaKTheory {
    /// Prime $p \ge 2$.
    pub prime_p: u64,
    /// Chromatic formal group height $h \ge 1$.
    pub height_h: u32,
}

impl ChromaticMoravaKTheory {
    /// Creates a new Morava $K(h)$ chromatic homotopy engine.
    pub fn new(prime_p: u64, height_h: u32) -> Result<Self> {
        if prime_p < 2 || height_h == 0 || height_h > 15 {
            return Err(Error::Msg(
                "prime_p >= 2 and height_h in 1..=15 required".to_string(),
            ));
        }
        Ok(Self { prime_p, height_h })
    }

    /// Computes the topological degree $|v_h| = 2(p^h - 1)$ of the $v_h$-periodicity generator and Morava stabilizer rank $h^2$.
    pub fn vn_generator_degree_and_stabilizer_dim(&self) -> (u64, u32) {
        (
            2 * (self.prime_p.pow(self.height_h) - 1),
            self.height_h * self.height_h,
        )
    }
}
