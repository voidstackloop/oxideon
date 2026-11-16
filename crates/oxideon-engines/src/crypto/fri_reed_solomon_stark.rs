//! Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI) & zk-STARK Split-and-Fold Degree Halving (Ben-Sasson et al., 2018).
//!
//! ### 1. FRI Split-and-Fold Operator $\mathcal{F}_\alpha[f](x^2)$
//! Decomposition of a degree-$< 2d$ polynomial $f(X) = g(X^2) + X h(X^2)$ into even and odd parts allows the verifier's
//! random challenge $\alpha \in \mathbb{F}_p$ to fold $f$ into a degree-$< d$ polynomial $f^\star(Y) = g(Y) + \alpha h(Y)$ via:
//! $$f^\star(x^2) = \frac{f(x) + f(-x)}{2} + \alpha \cdot \frac{f(x) - f(-x)}{2 x} \pmod p$$
//!
//! **Not a proof system:** this module only implements one fold step. There are no Merkle commitments, queries or
//! verifier. See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};

/// Fast Reed-Solomon IOP of Proximity (FRI) Split-and-Fold Engine.
#[derive(Debug, Clone, Copy)]
pub struct FastReedSolomonIopEngine {
    /// Prime field modulus $p \ge 3$.
    pub modulus_p: i64,
}

impl FastReedSolomonIopEngine {
    /// Creates a new FRI Split-and-Fold engine over $\mathbb{F}_p$.
    pub fn new(modulus_p: i64) -> Result<Self> {
        if modulus_p < 3 || modulus_p % 2 == 0 {
            return Err(Error::Msg(
                "modulus_p must be an odd prime >= 3".to_string(),
            ));
        }
        Ok(Self { modulus_p })
    }

    /// Folds the polynomial coefficient vector `[c_0, c_1, ..., c_{2d-1}]` of length $2d$ into `[c_0 + alpha*c_1, c_2 + alpha*c_3, ...]` of length $d$ mod $p$.
    pub fn fold_polynomial_coeffs(&self, coeffs: &[i64], challenge_alpha: i64) -> Result<Vec<i64>> {
        if coeffs.is_empty() || coeffs.len() % 2 != 0 {
            return Err(Error::Msg(
                "coeffs length must be positive and even".to_string(),
            ));
        }
        let p = self.modulus_p;
        if p < 3 {
            return Err(Error::Msg("modulus_p must be >= 3".to_string()));
        }
        let d = coeffs.len() / 2;
        let mut folded = Vec::with_capacity(d);
        // i128 products: alpha and the coefficients are unbounded i64s.
        for i in 0..d {
            let val = ((coeffs[2 * i] as i128
                + challenge_alpha as i128 * coeffs[2 * i + 1] as i128)
                .rem_euclid(p as i128)) as i64;
            folded.push(val);
        }
        Ok(folded)
    }
}
