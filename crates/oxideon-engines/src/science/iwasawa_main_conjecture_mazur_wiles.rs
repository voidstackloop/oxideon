//! Iwasawa Theory Class Number Formula & Mazur-Wiles Main Conjecture over $\mathbb{Z}_p$-Extensions (Iwasawa, 1959; Mazur & Wiles, 1984).
//!
//! ### 1. Iwasawa Asymptotic $p$-Class Number Formula $e_n = \mu p^n + \lambda n + \nu$
//! In a $\mathbb{Z}_p$-extension $K_\infty / K = \bigcup_{n \ge 0} K_n$, if the $p$-Sylow subgroup of the ideal class group of $K_n$ has order $p^{e_n}$, then for all sufficiently large $n$:
//! $$e_n = \mu p^n + \lambda n + \nu$$
//! By the Ferrero-Washington theorem, $\mu = 0$ for cyclotomic $\mathbb{Z}_p$-extensions of abelian number fields.

use crate::error::{Error, Result};

/// Iwasawa Class Number Growth & Main Conjecture Engine.
#[derive(Debug, Clone, Copy)]
pub struct IwasawaClassNumberGrowth {
    /// Prime $p \ge 2$.
    pub prime_p: u64,
    /// Iwasawa $\mu$-invariant ($\ge 0$).
    pub mu_inv: u64,
    /// Iwasawa $\lambda$-invariant ($\ge 0$).
    pub lambda_inv: u64,
    /// Iwasawa $\nu$-invariant ($\in \mathbb{Z}$).
    pub nu_inv: i64,
}

impl Default for IwasawaClassNumberGrowth {
    fn default() -> Self {
        Self {
            prime_p: 37,
            mu_inv: 0,
            lambda_inv: 1,
            nu_inv: 0,
        }
    }
}

impl IwasawaClassNumberGrowth {
    /// Creates a new Iwasawa Class Number Growth engine.
    pub fn new(prime_p: u64, mu_inv: u64, lambda_inv: u64, nu_inv: i64) -> Result<Self> {
        if prime_p < 2 {
            return Err(Error::Msg("prime_p must be >= 2".to_string()));
        }
        Ok(Self {
            prime_p,
            mu_inv,
            lambda_inv,
            nu_inv,
        })
    }

    /// Evaluates the exact $p$-adic valuation $e_n = \mu p^n + \lambda n + \nu$ of $|\operatorname{Cl}(K_n)[p^\infty]|$ at layer $n$.
    pub fn class_group_p_valuation_at_layer(&self, layer_n: u32) -> i64 {
        let pn = (self.prime_p as i64).saturating_pow(layer_n);
        (self.mu_inv as i64) * pn + (self.lambda_inv as i64) * (layer_n as i64) + self.nu_inv
    }
}
