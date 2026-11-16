//! Goodstein Hereditary Base-Change Theorem & Kirby-Paris $\varepsilon_0$ Proof-Theoretic Independence (Goodstein, 1944; Kirby & Paris, 1982).
//!
//! ### 1. Hereditary Base Bump & Fast-Growing Hierarchy $f_k(n)$
//! Writing $m$ in hereditary base $b$, replacing every $b$ by $b+1$, and subtracting $1$ terminates at $0$ for every initial $m$,
//! but this fact is independent of Peano Arithmetic (PA) because its termination ordinal is $\varepsilon_0 = \sup \{\omega, \omega^\omega, \omega^{\omega^\omega}, \dots\}$.

use crate::error::{Error, Result};

/// Goodstein Hereditary Base-Change & Proof-Theoretic Fast-Growing Hierarchy Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoodsteinParisHarringtonEngine;

impl GoodsteinParisHarringtonEngine {
    /// Creates a new Goodstein & Paris-Harrington engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates one hereditary base bump $G(m, b \to b+1)$ and the Goodstein step $G(m, b \to b+1) - 1$.
    pub fn hereditary_base_bump_and_step(m: u64, base_b: u64) -> Result<(u64, u64)> {
        if base_b < 2 {
            return Err(Error::Msg("base_b must be >= 2".to_string()));
        }
        fn bump(val: u64, b: u64, next_b: u64) -> Result<u64> {
            if val < b {
                return Ok(val);
            }
            let mut rem = val;
            let mut exp = 0u64;
            let mut acc = 0u64;
            while rem > 0 {
                let digit = rem % b;
                rem /= b;
                if digit > 0 {
                    let bumped_exp = bump(exp, b, next_b)?;
                    let p = next_b
                        .checked_pow(bumped_exp as u32)
                        .ok_or_else(|| Error::Msg("Goodstein tower overflowed u64".to_string()))?;
                    acc = acc
                        .checked_add(digit.saturating_mul(p))
                        .ok_or_else(|| Error::Msg("Goodstein sum overflowed u64".to_string()))?;
                }
                exp += 1;
            }
            Ok(acc)
        }
        let bumped = bump(m, base_b, base_b + 1)?;
        Ok((bumped, bumped.saturating_sub(1)))
    }
}
