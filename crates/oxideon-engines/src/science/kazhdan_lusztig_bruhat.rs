//! Kazhdan-Lusztig Polynomials $P_{x, w}(q)$, Iwahori-Hecke Algebra & Schubert Singularity Locus (Kazhdan & Lusztig, 1979; Elias & Williamson, 2014).
//!
//! ### 1. Iwahori-Hecke Quadratic Relation & Kazhdan-Lusztig Degree Bound
//! In the Iwahori-Hecke algebra $\mathcal{H}(W)$ over $\mathbb{Z}[q^{1/2}, q^{-1/2}]$, generators satisfy $(T_s + 1)(T_s - q) = 0$,
//! and Kazhdan-Lusztig polynomials satisfy $\deg P_{x, w}(q) \le \frac{1}{2}(\ell(w) - \ell(x) - 1)$ for $x < w$, with $P_{x, w}(1) = 1$ iff the Schubert variety $X_w$ is rationally smooth at $x$.

use crate::error::{Error, Result};

/// Kazhdan-Lusztig Polynomial & Iwahori-Hecke Algebra Engine.
#[derive(Debug, Clone, Copy)]
pub struct KazhdanLusztigHeckeEngine {
    /// Hecke specialization parameter $q > 0$.
    pub hecke_q: f64,
}

impl Default for KazhdanLusztigHeckeEngine {
    fn default() -> Self {
        Self { hecke_q: 2.0 }
    }
}

impl KazhdanLusztigHeckeEngine {
    /// Creates a new Kazhdan-Lusztig Hecke engine.
    pub fn new(hecke_q: f64) -> Result<Self> {
        if hecke_q <= 0.0 || !hecke_q.is_finite() {
            return Err(Error::Msg("hecke_q must be > 0".to_string()));
        }
        Ok(Self { hecke_q })
    }

    /// Evaluates the maximum allowed Kazhdan-Lusztig degree $\lfloor (\ell(w) - \ell(x) - 1)/2 \rfloor$ and the Poincaré polynomial $\sum_{w \in S_n} q^{\ell(w)} = \prod_{k=1}^n \[k\]_q$ of $S_n$.
    pub fn max_kl_degree_and_poincare_polynomial(
        &self,
        len_x: usize,
        len_w: usize,
        rank_n: usize,
    ) -> Result<(usize, f64)> {
        if len_x >= len_w || rank_n == 0 {
            return Err(Error::Msg(
                "len_x < len_w and rank_n >= 1 required".to_string(),
            ));
        }
        let max_deg = (len_w - len_x - 1) / 2;
        let mut poincare = 1.0;
        for k in 1..=rank_n {
            let q_int: f64 = (0..k).map(|j| self.hecke_q.powi(j as i32)).sum();
            poincare *= q_int;
        }
        Ok((max_deg, poincare))
    }
}
