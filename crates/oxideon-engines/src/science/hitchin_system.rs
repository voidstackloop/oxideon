//! Hitchin Integrable System, Higgs Bundles $(E, \Phi)$, Spectral Curves & Non-Abelian Hodge Correspondence (Hitchin, 1987).
//!
//! ### 1. Hitchin Self-Duality Equations on a Riemann Surface $\Sigma_g$
//! A Higgs bundle $(E, \Phi)$ with Higgs field $\Phi \in H^0(\Sigma, \mathrm{End}(E) \otimes K_\Sigma)$ satisfies Hitchin's
//! dimensional reduction of the 4D Self-Dual Yang-Mills equations:
//! $$F_A + [\Phi, \Phi^*] = 0, \qquad \bar{\partial}_A \Phi = 0$$
//!
//! ### 2. Hitchin Fibration & Spectral Curve Genus
//! For a rank-$r$ Higgs bundle over a curve $\Sigma$ of genus $g \ge 2$, the Hitchin base has complex dimension
//! $\dim \mathcal{B} = (r^2 - 1)(g - 1)$ for $SL(r, \mathbb{C})$ (or $r^2(g - 1) + 1$ for $GL(r, \mathbb{C})$),
//! equal to the genus of the **Hitchin Spectral Curve** $\det(\lambda I - \Phi) = 0$ inside $T^* \Sigma$.

use crate::error::{Error, Result};

/// Hitchin Integrable System, Spectral Curve & Higgs Bundle Engine.
#[derive(Debug, Clone, Copy)]
pub struct HitchinHiggsBundleEngine {
    /// Base Riemann surface genus $g \ge 2$.
    pub base_genus_g: usize,
    /// Vector bundle rank $r \ge 2$.
    pub bundle_rank_r: usize,
}

impl HitchinHiggsBundleEngine {
    /// Constructs a Hitchin system engine for rank $r \ge 2$ over genus $g \ge 2$.
    pub fn new(base_genus_g: usize, bundle_rank_r: usize) -> Result<Self> {
        if base_genus_g < 2 || bundle_rank_r < 2 {
            return Err(Error::Msg(
                "Hitchin system requires genus g >= 2 and rank r >= 2".to_string(),
            ));
        }
        Ok(Self {
            base_genus_g,
            bundle_rank_r,
        })
    }

    /// Computes the genus of the $GL(r, \mathbb{C})$ Hitchin spectral curve $\tilde{\Sigma} \subset T^* \Sigma$:
    /// $g(\tilde{\Sigma}) = r^2 (g - 1) + 1$ (which equals $\frac{1}{2} \dim_{\mathbb{C}} \mathcal{M}_{\text{Higgs}}$).
    pub fn spectral_curve_genus(&self) -> usize {
        self.bundle_rank_r * self.bundle_rank_r * (self.base_genus_g - 1) + 1
    }

    /// Evaluates the Poisson-commuting Hitchin Hamiltonians $(\operatorname{Tr}(\Phi), \frac{1}{2}\operatorname{Tr}(\Phi^2), \det(\Phi))$
    /// for a $2 \times 2$ Higgs matrix $\Phi = [[a, b], [c, d]]$.
    pub fn evaluate_sl2_hitchin_hamiltonians(phi: [f64; 4]) -> (f64, f64, f64) {
        let [a, b, c, d] = phi;
        let tr = a + d;
        let det = a * d - b * c;
        let half_tr_sq = 0.5 * (a * a + 2.0 * b * c + d * d);
        (tr, half_tr_sq, det)
    }
}
