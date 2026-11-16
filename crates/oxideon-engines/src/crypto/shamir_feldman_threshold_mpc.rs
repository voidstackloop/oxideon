//! Shamir $(t, n)$-Threshold Secret Sharing over $\mathbb{F}_p$ & Lagrange Barycentric Reconstruction (Shamir, 1979).
//!
//! ### 1. Information-Theoretic Perfect Secrecy & Lagrange Interpolation at $x = 0$
//! Encoding a secret $s = P(0) \in \mathbb{F}_p$ into a degree-$< t$ polynomial $P(X)$ evaluated at distinct points $x_i \in \mathbb{F}_p^\times$
//! allows any $t$ shares $(x_i, y_i)$ to reconstruct $s$ exactly via Lagrange interpolation at $X = 0$:
//! $$s = P(0) = \sum_{i=1}^t y_i \prod_{j \ne i} \frac{-x_j}{x_i - x_j} \pmod p$$
//!
//! **Not production crypto:** only reconstruction is provided (no share generation, no Feldman commitments),
//! over plain `i64` arithmetic that is not constant-time and overflows for large `p`.
//! See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};

/// Shamir $(t, n)$-Threshold Secret Sharing & MPC Reconstruction Engine.
#[derive(Debug, Clone, Copy)]
pub struct ShamirSecretSharingFp {
    /// Prime field modulus $p \ge 5$.
    pub modulus_p: i64,
}

impl ShamirSecretSharingFp {
    /// Creates a new Shamir Secret Sharing engine over $\mathbb{F}_p$.
    pub fn new(modulus_p: i64) -> Result<Self> {
        if modulus_p < 5 {
            return Err(Error::Msg("modulus_p must be a prime >= 5".to_string()));
        }
        Ok(Self { modulus_p })
    }

    fn mod_pow(mut base: i64, mut exp: i64, p: i64) -> i64 {
        let mut res = 1_i64;
        base = base.rem_euclid(p);
        while exp > 0 {
            if exp % 2 == 1 {
                res = (res * base).rem_euclid(p);
            }
            base = (base * base).rem_euclid(p);
            exp /= 2;
        }
        res
    }

    /// Reconstructs the secret $s = P(0) \pmod p$ from $t$ points `(xs, ys)` via Fermat modular inversion.
    pub fn reconstruct_secret(&self, xs: &[i64], ys: &[i64]) -> Result<i64> {
        if xs.is_empty() || xs.len() != ys.len() {
            return Err(Error::Msg(
                "Non-empty equal-length xs and ys required".to_string(),
            ));
        }
        let p = self.modulus_p;
        let t = xs.len();
        // Duplicate x coordinates (mod p) zero a Lagrange denominator and
        // silently reconstruct the wrong secret.
        {
            let mut seen = std::collections::HashSet::with_capacity(t);
            for &x in xs {
                if !seen.insert(x.rem_euclid(p)) {
                    return Err(Error::Msg(
                        "Shamir reconstruction needs distinct share x-coordinates".to_string(),
                    ));
                }
            }
        }
        let mut secret = 0_i64;
        for i in 0..t {
            let mut num = 1_i64;
            let mut den = 1_i64;
            for j in 0..t {
                if i != j {
                    num = (num * (-xs[j])).rem_euclid(p);
                    den = (den * (xs[i] - xs[j])).rem_euclid(p);
                }
            }
            let inv_den = Self::mod_pow(den, p - 2, p);
            let lagrange_0 = (num * inv_den).rem_euclid(p);
            secret = (secret + ys[i] * lagrange_0).rem_euclid(p);
        }
        Ok(secret)
    }
}
