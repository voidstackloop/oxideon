//! Seiberg-Witten 4-Manifold Monopole Equations, Virtual Moduli Dimension & Taubes Adjunction Inequality (Witten, 1994).
//!
//! ### 1. Seiberg-Witten Moduli Space Virtual Dimension $d(\mathfrak{s})$
//! For a $\mathrm{Spin}^c$ structure $\mathfrak{s}$ with determinant line bundle $L$ ($c_1(L)^2 = c_1^2$) on a closed
//! oriented 4-manifold $X$ with Euler characteristic $\chi(X)$ and signature $\sigma(X)$, the virtual dimension of
//! the Seiberg-Witten monopole moduli space is:
//! $$d(\mathfrak{s}) = \frac{1}{4}\left(c_1(L)^2 - (2 \chi(X) + 3 \sigma(X))\right)$$

/// Seiberg-Witten 4-Manifold Monopole Moduli & Adjunction Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeibergWittenMonopoleEngine;

impl SeibergWittenMonopoleEngine {
    /// Creates a new Seiberg-Witten monopole engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the virtual dimension $d(\mathfrak{s}) = \frac{1}{4}(c_1(L)^2 - 2\chi(X) - 3\sigma(X))$ of the Seiberg-Witten moduli space.
    pub fn moduli_virtual_dimension(c1_sq: i64, euler_chi: i64, signature_sigma: i64) -> f64 {
        0.25 * ((c1_sq - 2 * euler_chi - 3 * signature_sigma) as f64)
    }

    /// Verifies the Kronheimer-Mrowka-Thom-Morgan Seiberg-Witten Adjunction Inequality for a smoothly embedded surface
    /// $\Sigma \subset X$ of genus $g(\Sigma) \ge 1$ with self-intersection $[\Sigma]^2 \ge 0$:
    /// $2 g(\Sigma) - 2 \ge [\Sigma]^2 + |\langle c_1(L), [\Sigma] \rangle|$.
    pub fn satisfies_adjunction_inequality(
        genus_g: i64,
        self_intersection: i64,
        c1_pairing: i64,
    ) -> bool {
        if genus_g < 1 || self_intersection < 0 {
            return false;
        }
        (2 * genus_g - 2) >= self_intersection + c1_pairing.abs()
    }
}
