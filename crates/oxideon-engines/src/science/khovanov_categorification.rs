//! Khovanov Homology Categorification of the Jones Polynomial & Rasmussen Slice $s$-Invariant (Khovanov, 2000; Rasmussen, 2010).
//!
//! ### 1. Rasmussen's Combinatorial Proof of the Milnor Conjecture via Khovanov-Lee Homology
//! Khovanov homology $\mathrm{Kh}^{i,j}(K)$ bigradedly categorifies the Jones polynomial $J(K; q) = \sum_{i,j} (-1)^i q^j \dim \mathrm{Kh}^{i,j}(K)$.
//! Rasmussen's concordance invariant $s(K) \in 2\mathbb{Z}$ extracted from Lee's deformation bounds the smooth 4-ball slice genus
//! $|s(K)| \le 2 g_4(K)$, and for any positive torus knot $T_{p,q}$ ($\gcd(p, q) = 1$):
//! $$s(T_{p,q}) = (p - 1)(q - 1) \quad \Longrightarrow \quad g_4(T_{p,q}) = \frac{(p - 1)(q - 1)}{2}$$

use crate::error::{Error, Result};

/// Khovanov Categorification & Rasmussen Slice $s$-Invariant Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct KhovanovHomologyEngine;

impl KhovanovHomologyEngine {
    /// Creates a new Khovanov Homology engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes Rasmussen's invariant $s(T_{p,q}) = (p - 1)(q - 1)$ and the smooth 4-ball slice genus $g_4(T_{p,q}) = (p-1)(q-1)/2$ for a torus knot $T_{p,q}$.
    pub fn torus_knot_rasmussen_s_and_slice_genus(p: u64, q: u64) -> Result<(u64, u64)> {
        if p < 2 || q < 2 {
            return Err(Error::Msg(
                "Torus knot parameters p, q must be >= 2".to_string(),
            ));
        }
        let s = (p - 1) * (q - 1);
        Ok((s, s / 2))
    }
}
