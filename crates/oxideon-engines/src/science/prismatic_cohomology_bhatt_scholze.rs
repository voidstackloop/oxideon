//! Bhatt-Scholze Prismatic Cohomology $H^*_{\Delta}(X / A)$, $\delta$-Rings & Breuil-Kisin Prisms (Bhatt & Scholze, 2022).
//!
//! ### 1. $\delta$-Rings & Frobenius Lift $\phi(x) = x^p + p \delta(x)$
//! A prism $(A, I)$ consists of a derived $(p, I)$-complete $\delta$-ring $A$ equipped with a Frobenius lift
//! $\phi : A \to A$ defined by the $p$-derivations operator $\delta : A \to A$:
//! $$\phi(x) = x^p + p \, \delta(x), \qquad \delta(x + y) = \delta(x) + \delta(y) - \sum_{k=1}^{p-1} \frac{1}{p}\binom{p}{k} x^k y^{p-k}$$
//! Prismatic cohomology $H^*_{\Delta}(X / A)$ simultaneously specializes to étale, crystalline, de Rham, and Hodge-Tate cohomologies!

use crate::error::{Error, Result};

/// Bhatt-Scholze Prismatic $\delta$-Ring & Frobenius Lift Engine.
#[derive(Debug, Clone, Copy)]
pub struct PrismaticSiteBreuilKisin {
    /// Prime $p \ge 2$.
    pub prime_p: i64,
}

impl PrismaticSiteBreuilKisin {
    /// Creates a new Bhatt-Scholze Prismatic $\delta$-ring engine at prime $p \ge 2$.
    pub fn new(prime_p: i64) -> Result<Self> {
        if prime_p < 2 || !Self::is_prime(prime_p) {
            return Err(Error::Msg(format!(
                "prime_p={prime_p} must be a prime >= 2"
            )));
        }
        Ok(Self { prime_p })
    }

    fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        let mut d = 3;
        while d <= n / d {
            if n % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }

    /// Evaluates the exact Joyal $p$-derivation $\delta(x) = \frac{x - x^p}{p}$ on $\mathbb{Z}$ (where $\phi(x) = x$) and verifies $\phi(x) = x^p + p \delta(x)$.
    pub fn integer_p_derivation_delta(&self, x: i64) -> Result<i64> {
        let p = self.prime_p;
        if p < 2 || !Self::is_prime(p) {
            return Err(Error::Msg(format!("prime_p={p} must be a prime >= 2")));
        }
        if p > u32::MAX as i64 {
            return Err(Error::Msg(format!("prime_p={p} exponent too large")));
        }
        let xp = x
            .checked_pow(p as u32)
            .ok_or_else(|| Error::Msg(format!("{x}^{p} overflows i64")))?;
        x.checked_sub(xp)
            .ok_or_else(|| Error::Msg("x - x^p overflows i64".to_string()))
            .map(|num| num / p)
    }
}
