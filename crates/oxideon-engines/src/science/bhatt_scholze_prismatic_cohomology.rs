//! Bhatt-Scholze Prismatic Cohomology $H^*_{\Delta}(X / A)$ & Breuil-Kisin / $A_{\mathrm{inf}}$ Specialization (Bhatt & Scholze, 2019/2022).
//!
//! ### 1. Hodge-Tate Comparison $\Delta_{X/A} / I \cong \bigoplus_i \Omega^i_{X / (A/I)}\{-i\}$ & $q$-De Rham Specialization
//! Over a bounded prism $(A, I)$, prismatic cohomology $\Delta_{X / A}$ unifies crystalline, étale, and de Rham cohomology, with Hodge-Tate graded pieces isomorphic to twisted differential forms of total dimension $\sum_{i=0}^d \binom{d}{i} = 2^d$ for a smooth $d$-dimensional torus.

use crate::error::{Error, Result};

/// Bhatt-Scholze Prismatic Cohomology & $q$-De Rham Specialization Engine.
#[derive(Debug, Clone, Copy)]
pub struct BhattScholzePrismaticEngine {
    /// Smooth formal scheme dimension $d \ge 1$.
    pub smooth_dim_d: u32,
}

impl Default for BhattScholzePrismaticEngine {
    fn default() -> Self {
        Self { smooth_dim_d: 3 }
    }
}

impl BhattScholzePrismaticEngine {
    /// Creates a new Bhatt-Scholze Prismatic Cohomology engine.
    pub fn new(smooth_dim_d: u32) -> Result<Self> {
        if smooth_dim_d == 0 || smooth_dim_d > 30 {
            return Err(Error::Msg("smooth_dim_d must be in 1..=30".to_string()));
        }
        Ok(Self { smooth_dim_d })
    }

    /// Evaluates the Hodge-Tate exterior rank $2^d$ on a smooth $d$-torus and the $q$-integer $\[n\]_q = (q^n - 1) / (q - 1)$ of $q$-de Rham cohomology.
    pub fn hodge_tate_rank_and_q_integer(&self, n_int: u32, q_param: f64) -> Result<(u64, f64)> {
        if !q_param.is_finite() || n_int == 0 {
            return Err(Error::Msg(
                "q_param finite and n_int >= 1 required".to_string(),
            ));
        }
        let rank = 1_u64 << self.smooth_dim_d;
        let q_n = if (q_param - 1.0).abs() < 1e-12 {
            n_int as f64
        } else {
            (q_param.powi(n_int as i32) - 1.0) / (q_param - 1.0)
        };
        Ok((rank, q_n))
    }
}
