//! Adams Spectral Sequence, Stable Homotopy Groups of Spheres $\pi_n^s$ & Adams-Quillen $J$-Homomorphism (Adams, 1966).
//!
//! ### 1. Adams-Quillen Theorem on the Image of the Stable $J$-Homomorphism
//! The stable $J$-homomorphism $J_m : \pi_m(\mathrm{SO}) \to \pi_m^s(\mathbb{S}^0)$ detects the $v_1$-periodic
//! primary family in the stable homotopy groups of spheres. For $m = 4k - 1$, Adams proved that $\operatorname{Im}(J_{4k-1})$
//! is a cyclic direct summand of $\pi_{4k-1}^s$ of exact order:
//! $$\left|\operatorname{Im}(J_{4k-1})\right| = \operatorname{denom}\left(\frac{B_{2k}}{4 k}\right)$$
//! For $k = 1, 2, 3$ ($m = 3, 7, 11$), this yields the famous orders $24, 240, 504$!

use crate::error::{Error, Result};

/// Adams Spectral Sequence & Stable $J$-Homomorphism Order Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdamsJHomomorphismEngine;

impl AdamsJHomomorphismEngine {
    /// Creates a new Adams $J$-Homomorphism engine.
    pub fn new() -> Self {
        Self
    }

    /// Returns the exact order $|\operatorname{Im}(J_{4k-1})| = \operatorname{denom}(B_{2k} / 4k)$ for $k \in 1..=6$.
    pub fn image_of_j_order(index_k: usize) -> Result<u64> {
        match index_k {
            1 => Ok(24),    // pi_3^s = Z/24
            2 => Ok(240),   // pi_7^s = Z/240
            3 => Ok(504),   // Im(J_11) = Z/504
            4 => Ok(480),   // Im(J_15) = Z/480
            5 => Ok(264),   // Im(J_19) = Z/264
            6 => Ok(65520), // Im(J_23) = Z/65520
            _ => Err(Error::Msg("index_k must be in 1..=6".to_string())),
        }
    }
}
