//! Shelah Model-Theoretic NIP (No Independence Property) & Sauer-Shelah-Vapnik-Chervonenkis Lemma (Shelah, 1972; Sauer, 1972).
//!
//! ### 1. Sauer-Shelah Polynomial Growth Bound for NIP Formulas
//! A first-order formula $\varphi(x; y)$ has NIP iff its VC dimension $d < \infty$. Over a finite parameter set of size $n \ge d$,
//! the number of $\varphi$-types $|\mathcal{S}_\varphi(A)|$ is bounded by the polynomial growth function:
//! $$\Phi_d(n) = \sum_{k=0}^{d} \binom{n}{k}$$

use crate::error::{Error, Result};

/// Shelah Model-Theoretic NIP Stability & Sauer-Shelah Type Counting Engine.
#[derive(Debug, Clone, Copy)]
pub struct ShelahStabilitySauerEngine {
    /// VC dimension / alternation rank $d$ of the NIP formula $\varphi(x; y)$.
    pub vc_dim_d: usize,
}

impl Default for ShelahStabilitySauerEngine {
    fn default() -> Self {
        Self { vc_dim_d: 3 }
    }
}

impl ShelahStabilitySauerEngine {
    /// Creates a new Shelah NIP & Sauer-Shelah engine.
    pub fn new(vc_dim_d: usize) -> Self {
        Self { vc_dim_d }
    }

    /// Evaluates the exact Sauer-Shelah bound $\Phi_d(n) = \sum_{k=0}^d \binom{n}{k}$ on the number of $\varphi$-types over $n$ parameters.
    pub fn sauer_shelah_type_bound(&self, num_params_n: usize) -> Result<u64> {
        if num_params_n > 60 {
            return Err(Error::Msg(
                "num_params_n must be <= 60 to avoid overflow".to_string(),
            ));
        }
        let k_max = self.vc_dim_d.min(num_params_n);
        let mut total = 0u64;
        let mut binom = 1u64;
        for k in 0..=k_max {
            total += binom;
            if k < k_max {
                binom = binom * ((num_params_n - k) as u64) / ((k + 1) as u64);
            }
        }
        Ok(total)
    }
}
