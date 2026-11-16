//! Selberg $\Lambda^2$-Sieve, Bombieri-Vinogradov Prime Distribution & Maynard-Tao Bounded Prime Gaps (Selberg, 1947; Maynard, 2015).
//!
//! ### 1. Selberg's $\Lambda^2$-Sieve Quadratic Optimization
//! To upper-bound the number of integers $n \le X$ free of prime factors $p < z$, Selberg replaces the Möbius function
//! with squared real weights $\left(\sum_{d \mid n, d < z} \lambda_d\right)^2$ ($\lambda_1 = 1$), whose minimum is governed
//! by the sum $J(z) = \sum_{d < z} \frac{\mu^2(d)}{\varphi(d)} \approx \ln z$.

use crate::error::{Error, Result};

/// Selberg $\Lambda^2$-Sieve & Maynard-Tao Prime Tuple Sieve Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SelbergSieveEngine;

impl SelbergSieveEngine {
    /// Creates a new Selberg Sieve engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes Selberg's sieve denominator $J(z) = \sum_{d=1}^{z} \frac{\mu^2(d)}{\varphi(d)}$ over squarefree integers $d \le z$.
    pub fn selberg_j_sum(z_limit: usize) -> Result<f64> {
        if z_limit == 0 {
            return Err(Error::Msg("z_limit must be >= 1".to_string()));
        }
        // Trial division per d: about z_limit^1.5 operations, so this bound is lower than the generic iteration limit.
        if z_limit > 10_000_000 {
            return Err(Error::Msg(format!(
                "z_limit = {z_limit} is above the limit of 10000000"
            )));
        }
        let mut j_sum = 0.0;
        for d in 1..=z_limit {
            let mut n = d;
            let mut sq_free = true;
            let mut phi = 1_usize;
            let mut p = 2_usize;
            while p * p <= n {
                if n % p == 0 {
                    n /= p;
                    if n % p == 0 {
                        sq_free = false;
                        break;
                    }
                    phi *= p - 1;
                }
                p += 1;
            }
            if sq_free {
                if n > 1 {
                    phi *= n - 1;
                }
                j_sum += 1.0 / (phi as f64);
            }
        }
        Ok(j_sum)
    }
}
