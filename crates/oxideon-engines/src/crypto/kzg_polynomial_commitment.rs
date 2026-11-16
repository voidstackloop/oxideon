//! Kate-Zaverucha-Goldberg (KZG) Polynomial Commitments, Quotient Opening Witness $q(X) = \frac{f(X) - f(z)}{X - z}$ & EIP-4844 Blob Verification (Kate, Zaverucha & Goldberg, 2010).
//!
//! ### 1. Polynomial Evaluation Witness via Bezout's Factor Theorem over $\mathbb{F}_p$
//! For any polynomial $f(X) \in \mathbb{F}_p\[X\]$ and evaluation point $z \in \mathbb{F}_p$ with value $y = f(z)$,
//! $X - z$ divides $f(X) - y$ exactly in $\mathbb{F}_p\[X\]$, yielding the KZG opening quotient polynomial:
//! $$q(X) = \frac{f(X) - f(z)}{X - z}$$
//!
//! **Not a commitment scheme:** this module only divides polynomials over $\mathbb{F}_p$. There is no trusted setup,
//! group element or pairing check. See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};

/// Kate-Zaverucha-Goldberg (KZG) Quotient Witness & Evaluation Engine.
#[derive(Debug, Clone, Copy)]
pub struct KzgPolynomialCommitment {
    /// Prime field modulus $p \ge 3$.
    pub modulus_p: i64,
}

impl KzgPolynomialCommitment {
    /// Creates a new KZG Polynomial Commitment engine over $\mathbb{F}_p$.
    pub fn new(modulus_p: i64) -> Result<Self> {
        if modulus_p < 3 {
            return Err(Error::Msg("modulus_p must be >= 3".to_string()));
        }
        Ok(Self { modulus_p })
    }

    /// Computes `(evaluation_y, quotient_coeffs_low_to_high)` where $f(X) - y = q(X) (X - z) \pmod p$.
    pub fn compute_opening_quotient(
        &self,
        coeffs_low_to_high: &[i64],
        point_z: i64,
    ) -> Result<(i64, Vec<i64>)> {
        if coeffs_low_to_high.len() < 2 {
            return Err(Error::Msg("Polynomial degree must be >= 1".to_string()));
        }
        let p = self.modulus_p;
        if p < 3 {
            return Err(Error::Msg("modulus_p must be >= 3".to_string()));
        }
        let deg = coeffs_low_to_high.len() - 1;
        let mut q = vec![0_i64; deg];
        let mut rem = coeffs_low_to_high[deg].rem_euclid(p);
        // i128 Horner step: rem < p but point_z is an unbounded i64, so the
        // i64 product can overflow (debug panic / wrapped wrong quotient).
        for i in (0..deg).rev() {
            q[i] = rem;
            rem = ((coeffs_low_to_high[i] as i128 + rem as i128 * point_z as i128)
                .rem_euclid(p as i128)) as i64;
        }
        Ok((rem, q))
    }
}
