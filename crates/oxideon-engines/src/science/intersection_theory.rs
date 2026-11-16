//! Fulton-MacPherson Intersection Theory, Chow Rings $A^*(X)$ & Hirzebruch-Riemann-Roch Theorem.
//!
//! ### 1. Hirzebruch-Riemann-Roch Theorem (1954)
//! For a holomorphic vector bundle $E$ on a smooth projective manifold $X$ of complex dimension $n$,
//! the holomorphic Euler characteristic $\chi(X, E) = \sum_{i=0}^n (-1)^i \dim H^i(X, E)$ equals the top intersection pairing:
//! $$\chi(X, E) = \int_X \operatorname{ch}(E) \smile \operatorname{td}(T_X)$$
//! • On a curve $C$ of genus $g$ ($\operatorname{td}_1(C) = 1 - g$) and line bundle $\mathcal{L}$ of degree $d$:
//! $$\chi(C, \mathcal{L}) = \operatorname{rk}(E)(1 - g) + \deg(E)$$
//! • On a surface $S$ with first Chern class $c_1(S) = -K_S$ and topological Euler characteristic $c_2(S) = e(S)$:
//! $$\chi(S, \mathcal{O}_S(D)) = \chi(\mathcal{O}_S) + \frac{1}{2} D \cdot (D - K_S)$$

/// Fulton-MacPherson Chow Ring & Hirzebruch-Riemann-Roch Intersection Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChowHirzebruchRiemannRoch;

impl ChowHirzebruchRiemannRoch {
    /// Creates a new Chow & Hirzebruch-Riemann-Roch engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes $\chi(C, E) = \operatorname{rk}(E)(1 - g) + \deg(E)$ on a smooth projective curve of genus $g$.
    pub fn riemann_roch_curve(genus_g: i64, rank: i64, degree: i64) -> i128 {
        (rank as i128) * (1 - (genus_g as i128)) + (degree as i128)
    }

    /// Computes Noether's Formula $\chi(\mathcal{O}_S) = \frac{1}{12}(K_S^2 + c_2(S))$ and Riemann-Roch for a divisor $D$
    /// on an algebraic surface: $\chi(S, \mathcal{O}_S(D)) = \chi(\mathcal{O}_S) + \frac{1}{2}(D^2 - D \cdot K_S)$.
    pub fn riemann_roch_surface(
        k_self_intersection: i64,
        euler_c2: i64,
        d_self: i64,
        d_dot_k: i64,
    ) -> f64 {
        let chi_o = ((k_self_intersection as f64) + (euler_c2 as f64)) / 12.0;
        chi_o + 0.5 * ((d_self as f64) - (d_dot_k as f64))
    }

    /// Computes Bézout's intersection number $\deg(H_1 \cap \cdots \cap H_n) = \prod_{i=1}^n d_i$ in $A^*(\mathbb{P}^n)$.
    pub fn bezout_intersection_number(degrees: &[u64]) -> u64 {
        // Saturating: the product of absurd degrees is "at least u64::MAX", never a panic.
        degrees.iter().fold(1u64, |acc, &d| acc.saturating_mul(d))
    }
}
