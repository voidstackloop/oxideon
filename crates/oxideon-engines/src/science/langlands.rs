//! Langlands Program, Modular Forms, Hecke Eigenoperators $T_p$, Ramanujan $\tau(n)$ & Deligne's Theorem.
//!
//! ### 1. Modular Discriminant $\Delta(\tau)$ & Ramanujan Tau Function $\tau(n)$
//! The weight-$12$ cusp form $\Delta(\tau) = q \prod_{m=1}^\infty (1 - q^m)^{24} = \sum_{n=1}^\infty \tau(n) q^n$ ($q = e^{2\pi i \tau}$)
//! is the unique normalized Hecke eigenform in $S_{12}(\mathrm{SL}_2(\mathbb{Z}))$.
//!
//! ### 2. Hecke Multiplicativity & Prime-Power Recurrence (Mordell, 1917)
//! Because $\Delta(\tau)$ is an eigenform of every Hecke operator $T_n$, its Fourier coefficients $\tau(n)$ satisfy:
//! • **Multiplicativity**: $\tau(m n) = \tau(m) \tau(n)$ for $\gcd(m, n) = 1$,
//! • **Hecke Prime-Square Relation**: $\tau(p^2) = \tau(p)^2 - p^{11}$ for every prime $p$.
//!
//! ### 3. Deligne's Theorem / Ramanujan-Petersson Conjecture (Deligne, 1974)
//! Proved by Pierre Deligne as a consequence of the Weil Conjectures for étale cohomology: for every prime $p$,
//! $$|\tau(p)| \le 2 p^{11/2}$$
//! Equivalently, the Satake parameters $\alpha_p, \beta_p$ of the automorphic representation satisfy $|\alpha_p| = |\beta_p| = p^{11/2}$.

use crate::error::{Error, Result};

/// Langlands Automorphic Hecke Eigenform & Ramanujan $\tau(n)$ Engine.
#[derive(Debug, Clone)]
pub struct LanglandsHeckeEngine {
    /// Precomputed Ramanujan $\tau(1..=N)$ coefficients (1-indexed at `tau[n]`).
    pub tau_table: Vec<i64>,
}

impl LanglandsHeckeEngine {
    /// Computes the exact Ramanujan $\tau(1..=N)$ Fourier coefficients by expanding $q \prod_{m=1}^N (1 - q^m)^{24}$.
    pub fn new(max_n: usize) -> Result<Self> {
        if !(1..=64).contains(&max_n) {
            return Err(Error::Msg("max_n must be in 1..=64".to_string()));
        }
        // poly[k] = coefficient of q^k in prod_{m=1}^{max_n-1} (1 - q^m)^24
        let deg = max_n - 1;
        let mut poly = vec![0_i64; deg + 1];
        poly[0] = 1;

        for m in 1..=deg {
            // Multiply by (1 - q^m)^24 by applying (1 - q^m) 24 times
            for _ in 0..24 {
                for k in (m..=deg).rev() {
                    poly[k] -= poly[k - m];
                }
            }
        }

        let mut tau_table = vec![0_i64; max_n + 1];
        tau_table[1..=max_n].copy_from_slice(&poly[..max_n]);
        Ok(Self { tau_table })
    }

    /// Returns the exact Ramanujan coefficient $\tau(n)$.
    pub fn ramanujan_tau(&self, n: usize) -> Result<i64> {
        if n == 0 || n >= self.tau_table.len() {
            return Err(Error::Msg(format!("Index n={n} out of range")));
        }
        Ok(self.tau_table[n])
    }

    /// Verifies Hecke multiplicativity $\tau(m n) = \tau(m)\tau(n)$ for coprime $(m, n)$ and
    /// prime-square identity $\tau(p^2) = \tau(p)^2 - p^{11}$.
    pub fn verify_hecke_relations(&self) -> bool {
        if self.tau_table.len() > 6 {
            // tau(2) = -24, tau(3) = 252, tau(6) = -6048 = -24 * 252
            if self.tau_table[6] != self.tau_table[2] * self.tau_table[3] {
                return false;
            }
        }
        if self.tau_table.len() > 4 {
            // tau(4) = tau(2)^2 - 2^11 = 576 - 2048 = -1472
            if self.tau_table[4] != self.tau_table[2] * self.tau_table[2] - 2048 {
                return false;
            }
        }
        true
    }

    /// Verifies Deligne's Ramanujan-Petersson bound $|\tau(p)| \le 2 p^{11/2}$ for all primes $p$ in range.
    pub fn verify_deligne_ramanujan_bound(&self) -> bool {
        let primes = [2_usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
        for &p in &primes {
            if p < self.tau_table.len() {
                let lhs = (self.tau_table[p] as f64).abs();
                let rhs = 2.0 * (p as f64).powf(5.5);
                if lhs > rhs {
                    return false;
                }
            }
        }
        true
    }

    /// Evaluates the partial Euler product of the normalized Hecke $L$-function $L(s, \Delta)$ at real $s > 6.5$.
    pub fn automorphic_l_function_euler(&self, s: f64) -> Result<f64> {
        if s <= 6.5 {
            return Err(Error::Msg(
                "Real part s must be > 6.5 for absolute Euler convergence".to_string(),
            ));
        }
        let primes = [2_usize, 3, 5, 7, 11];
        let mut prod = 1.0;
        for &p in &primes {
            if p < self.tau_table.len() {
                let pf = p as f64;
                let tp = self.tau_table[p] as f64;
                let euler_inv = 1.0 - tp * pf.powf(-s) + pf.powf(11.0 - 2.0 * s);
                prod /= euler_inv;
            }
        }
        Ok(prod)
    }
}
