//! Iwasawa Theory of $\mathbb{Z}_p$-Extensions, $\Lambda$-Modules & Mazur-Wiles Main Conjecture (Iwasawa, 1959; Mazur & Wiles, 1984).
//!
//! ### 1. Iwasawa's Asymptotic Class Number Formula in a $\mathbb{Z}_p$-Tower
//! In a $\mathbb{Z}_p$-extension $K_\infty = \bigcup_{n \ge 0} K_n$ where $\operatorname{Gal}(K_n / K) \cong \mathbb{Z} / p^n \mathbb{Z}$,
//! let $p^{e_n}$ be the exact power of $p$ dividing the ideal class number $h(K_n)$. Iwasawa proved that for all $n \gg 0$:
//! $$e_n = \mu p^n + \lambda n + \nu$$
//! For cyclotomic $\mathbb{Z}_p$-extensions of abelian number fields, the Ferrero-Washington Theorem proves $\mu = 0$.

use crate::error::{Error, Result};

/// Iwasawa $\mathbb{Z}_p$-Tower Class Growth & $(\mu, \lambda, \nu)$ Invariant Engine.
#[derive(Debug, Clone, Copy)]
pub struct IwasawaMainConjectureEngine {
    /// Prime $p \ge 2$.
    pub prime_p: u64,
    /// Iwasawa $\mu$-invariant ($\mu \ge 0$, $\mu = 0$ by Ferrero-Washington for abelian fields).
    pub mu_inv: u64,
    /// Iwasawa $\lambda$-invariant ($\lambda \ge 0$).
    pub lambda_inv: u64,
    /// Iwasawa $\nu$-invariant ($\nu \in \mathbb{Z}$).
    pub nu_inv: i64,
}

impl IwasawaMainConjectureEngine {
    /// Creates a new Iwasawa $\mathbb{Z}_p$-extension engine.
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

    /// Computes the $p$-adic valuation $e_n = v_p(h(K_n)) = \mu p^n + \lambda n + \nu$ at layer $n$ of the $\mathbb{Z}_p$-tower.
    pub fn class_number_p_valuation(&self, layer_n: u32) -> i64 {
        let pn = self.prime_p.pow(layer_n) as i64;
        (self.mu_inv as i64) * pn + (self.lambda_inv as i64) * (layer_n as i64) + self.nu_inv
    }
}
