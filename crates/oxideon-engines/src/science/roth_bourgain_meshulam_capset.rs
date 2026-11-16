//! Roth-Meshulam Theorem & Croot-Lev-Pach-Ellenberg-Gijswijt Polynomial Method Cap-Set Bound (Roth, 1953; Ellenberg & Gijswijt, 2017).
//!
//! ### 1. Exponential Cap-Set Upper Bound in $\mathbb{F}_3^n$: $|A| \le 3 \cdot (2.7551)^n$
//! A subset $A \subset \mathbb{F}_3^n$ with no 3-term arithmetic progression $x + y + z = 0$ satisfies the Croot-Lev-Pach / Ellenberg-Gijswijt slice-rank bound:
//! $$|A| \le 3 \left(\frac{1 + \sqrt{33}}{8} \cdot 3\right)^n \approx 3 \cdot (2.755104)^n$$

use crate::error::{Error, Result};

/// Largest `n` whose counts and `3m` product still fit `u128` exactly (3^80 < u128::MAX).
const EXACT_MAX_N: usize = 79;

/// Roth-Meshulam & Ellenberg-Gijswijt Cap-Set Polynomial Method Engine.
#[derive(Debug, Clone, Copy)]
pub struct RothBourgainCapSetBounds {
    /// Dimension $n \ge 1$ of $\mathbb{F}_3^n$.
    pub dim_n: usize,
}

impl Default for RothBourgainCapSetBounds {
    fn default() -> Self {
        Self { dim_n: 10 }
    }
}

impl RothBourgainCapSetBounds {
    /// Creates a new Cap-Set Polynomial Method engine.
    pub fn new(dim_n: usize) -> Result<Self> {
        if dim_n == 0 {
            return Err(Error::Msg("dim_n must be >= 1".to_string()));
        }
        Ok(Self { dim_n })
    }

    /// Returns the Ellenberg-Gijswijt asymptotic base $\theta_3 = 2.75510461$ and an upper bound on a 3-AP-free cap set in $\mathbb{F}_3^n$.
    /// For $n \le 79$ the bound is the exact slice-rank bound $3m$, $m = \#\{a \in \{0,1,2\}^n : \sum a_i \le 2n/3\}$, capped by the trivial $3^n$;
    /// beyond that it is the closed-form relaxation $3\theta_3^n$ (always $\ge 3m$, and already below $3^n$ from $n = 13$).
    pub fn capset_growth_base_and_upper_bound(&self) -> (f64, f64) {
        let theta3: f64 = 2.7551046133430185;
        let n = self.dim_n;
        if n > EXACT_MAX_N {
            return (theta3, 3.0 * theta3.powf(n as f64));
        }
        let d = 2 * n / 3;
        // dp[s] = number of monomials of total degree s over the variables folded in so far.
        let mut dp = vec![0u128; d + 1];
        dp[0] = 1;
        for _ in 0..n {
            for s in (1..=d).rev() {
                dp[s] += dp[s - 1] + if s >= 2 { dp[s - 2] } else { 0 };
            }
        }
        let bound = (3 * dp.iter().sum::<u128>()).min(3u128.pow(n as u32));
        (theta3, bound as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(n: usize) -> f64 {
        RothBourgainCapSetBounds::new(n)
            .unwrap()
            .capset_growth_base_and_upper_bound()
            .1
    }

    /// Hand-counted 3m; n = 1..=3 are absorbed by the trivial 3^n cap (3, 9, 27).
    #[test]
    fn exact_small_values() {
        for (n, expected) in [
            (1, 3.0),
            (2, 9.0),
            (3, 27.0),
            (4, 45.0),
            (5, 153.0),
            (6, 504.0),
        ] {
            assert_eq!(bound(n), expected, "n={n}");
        }
    }

    /// The bound must stay at or above the known maximum cap-set sizes in F_3^n (n = 1..=6).
    #[test]
    fn bound_covers_known_maxima() {
        for (n, max_cap) in [
            (1, 2.0),
            (2, 4.0),
            (3, 9.0),
            (4, 20.0),
            (5, 45.0),
            (6, 112.0),
        ] {
            assert!(bound(n) >= max_cap, "n={n}: ub={}", bound(n));
        }
    }

    /// Past the exact range the closed form takes over: huge n returns instantly, u128 never overflows, and the bound never drops across the switch.
    #[test]
    fn large_n_is_cheap_and_monotone() {
        let theta: f64 = 2.7551046133430185;
        assert!(bound(EXACT_MAX_N) <= 3.0 * theta.powf(EXACT_MAX_N as f64));
        assert!(bound(EXACT_MAX_N + 1) >= bound(EXACT_MAX_N));
        assert!(bound(1_000_000_000) >= bound(EXACT_MAX_N + 1));
    }
}
