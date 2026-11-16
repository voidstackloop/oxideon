//! Atiyah-Singer Index Theorem, Dirac $\hat{A}$-Genus, Hirzebruch Signature & Rokhlin's Theorem (Atiyah & Singer, 1963).
//!
//! ### 1. Atiyah-Singer Index of the Twisted Spin Dirac Operator $D_E^+$ on a 4-Manifold
//! On a closed 4-dimensional Spin manifold $M$ coupled to a complex vector bundle $E$ of rank $r$, first Chern class $c_1(E)$,
//! and second Chern number $c_2(E)\[M\]$, the analytical Fredholm index $\operatorname{ind}(D_E^+) = \dim \ker D_E^+ - \dim \operatorname{coker} D_E^+$
//! equals the topological integral:
//! $$\operatorname{ind}(D_E^+) = \int_M \hat{A}(TM) \wedge \operatorname{ch}(E) = -\frac{r}{24} p_1(TM)\[M\] + \frac{1}{2} c_1(E)^2\[M\] - c_2(E)\[M\]$$
//! Using Hirzebruch's Signature Theorem $\sigma(M) = \frac{1}{3} p_1(TM)\[M\]$, the $\hat{A}$-genus is $\hat{A}(M) = -\frac{1}{8}\sigma(M)$
//! (which is an even integer by **Rokhlin's Theorem**: $\sigma(M) \equiv 0 \pmod{16}$!).

/// Atiyah-Singer Dirac Index & Hirzebruch-Rokhlin $\hat{A}$-Genus Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct AtiyahSingerIndexEngine;

impl AtiyahSingerIndexEngine {
    /// Creates a new Atiyah-Singer Index engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the $\hat{A}$-genus $\hat{A}(M) = -\sigma(M) / 8$ of a 4-dimensional Spin manifold from its signature $\sigma(M)$.
    pub fn a_hat_genus_4d(signature_sigma: i64) -> f64 {
        -(signature_sigma as f64) / 8.0
    }

    /// Computes the Atiyah-Singer analytical Dirac index $\operatorname{ind}(D_E^+) = r \hat{A}(M) + \frac{1}{2} c_1(E)^2 - c_2(E)$ on a Spin 4-manifold.
    pub fn twisted_dirac_index_4d(
        signature_sigma: i64,
        bundle_rank_r: i64,
        c1_sq: i64,
        c2: i64,
    ) -> f64 {
        (bundle_rank_r as f64) * Self::a_hat_genus_4d(signature_sigma) + 0.5 * (c1_sq as f64)
            - (c2 as f64)
    }
}
