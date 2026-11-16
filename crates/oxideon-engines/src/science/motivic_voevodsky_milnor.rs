//! Voevodsky $\mathbb{A}^1$-Motivic Homotopy, Milnor $K$-Theory Steinberg Relation & Norm Residue Theorem (Voevodsky, 2003 & 2011).
//!
//! ### 1. Milnor $K$-Theory Steinberg Symbol $\{a, 1 - a\} = 0$ & Hilbert Symbol Pairing
//! The Milnor $K$-ring $K_*^M(F) = T^*(F^\times) / \langle a \otimes (1 - a) \rangle$ is the source of the Galois norm residue
//! isomorphism $K_n^M(F) / m \xrightarrow{\sim} H^n_{\text{ét}}(F, \mu_m^{\otimes n})$ (proved by Voevodsky & Rost).
//! At $m = 2$, the degree-2 symbol $\{a, b\} \in K_2^M(\mathbb{Q}_p)/2$ is detected by the quadratic **Hilbert Symbol** $(a, b)_p \in \{-1, +1\}$.

use crate::error::{Error, Result};

/// Voevodsky Motivic Cohomology, Milnor $K$-Theory & Tate-Hilbert Symbol Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoevodskyMilnorBlochKato;

impl VoevodskyMilnorBlochKato {
    /// Creates a new Voevodsky-Milnor-Bloch-Kato engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the Legendre quadratic residue symbol $\left(\frac{a}{p}\right) \in \{-1, 0, +1\}$ for odd prime $p \ge 3$,
    /// which governs the tame Hilbert symbol in $K_2^M(\mathbb{Q}_p) / 2$.
    pub fn legendre_tame_symbol(a: i64, odd_prime_p: i64) -> Result<i32> {
        if odd_prime_p < 3 || odd_prime_p % 2 == 0 {
            return Err(Error::Msg(
                "odd_prime_p must be an odd prime >= 3".to_string(),
            ));
        }
        let rem = a.rem_euclid(odd_prime_p);
        if rem == 0 {
            return Ok(0);
        }
        let mut base = rem;
        let mut exp = (odd_prime_p - 1) / 2;
        let mut acc = 1_i64;
        while exp > 0 {
            if exp % 2 == 1 {
                acc = (acc * base) % odd_prime_p;
            }
            base = (base * base) % odd_prime_p;
            exp /= 2;
        }
        Ok(if acc == 1 { 1 } else { -1 })
    }
}
