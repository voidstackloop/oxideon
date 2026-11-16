//! Masser-Oesterlé $abc$-Conjecture, Radical $\operatorname{rad}(abc)$ & Szpiro's Discriminant-Conductor Ratio.
//!
//! ### 1. Masser-Oesterlé $abc$-Quality $q(a, b, c)$
//! For coprime positive integers $a + b = c$ ($\gcd(a, b) = 1$), let $\operatorname{rad}(n) = \prod_{p \mid n} p$ be the
//! squarefree radical. The $abc$-quality is defined as:
//! $$q(a, b, c) = \frac{\ln c}{\ln \operatorname{rad}(a b c)}$$
//! The $abc$-Conjecture asserts that for every $\epsilon > 0$, only finitely many coprime triples satisfy $q(a, b, c) > 1 + \epsilon$.

use crate::error::{Error, Result};

/// Masser-Oesterlé $abc$-Conjecture Quality & Squarefree Radical Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbcSzpiroArakelovEngine;

impl AbcSzpiroArakelovEngine {
    /// Creates a new $abc$-Szpiro-Arakelov engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the squarefree radical $\operatorname{rad}(n) = \prod_{p \mid n} p$ for $n \ge 1$.
    pub fn squarefree_radical(mut n: u64) -> u64 {
        if n <= 1 {
            return 1;
        }
        let mut rad = 1_u64;
        let mut d = 2_u64;
        while d * d <= n {
            if n % d == 0 {
                rad *= d;
                while n % d == 0 {
                    n /= d;
                }
            }
            d += 1;
        }
        if n > 1 {
            rad *= n;
        }
        rad
    }

    /// Evaluates the $abc$-quality $q(a, b, c) = \frac{\ln c}{\ln \operatorname{rad}(a b c)}$ for $a + b = c$.
    pub fn abc_quality(a: u64, b: u64) -> Result<f64> {
        if a == 0 || b == 0 {
            return Err(Error::Msg("a and b must be >= 1".to_string()));
        }
        let c = a + b;
        let rad_abc = Self::squarefree_radical(a * b * c);
        Ok((c as f64).ln() / (rad_abc as f64).ln())
    }
}
