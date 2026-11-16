//! Yau's Theorem on the Calabi Conjecture, Complex Monge-Ampère Equation & Ricci-Flat Kähler Metrics (Yau, 1978).
//!
//! ### 1. Complex Monge-Ampère Equation on a Kähler Manifold $(X, \omega)$
//! Finding a Ricci-flat Kähler metric $\omega_\phi = \omega + i \partial \bar{\partial} \phi$ in the same Kähler class $[\omega]$
//! reduces by $\partial\bar{\partial}$-lemma to Yau's **Complex Monge-Ampère Equation**:
//! $$\frac{(\omega + i \partial \bar{\partial} \phi)^n}{\omega^n} = \frac{\det(g_{i\bar{j}} + \partial_i \bar{\partial}_j \phi)}{\det(g_{i\bar{j}})} = e^F$$

use crate::error::{Error, Result};

/// Calabi-Yau Complex Monge-Ampère & Kähler Potential Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CalabiYauMongeAmpereEngine;

impl CalabiYauMongeAmpereEngine {
    /// Creates a new Calabi-Yau Monge-Ampère engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the $2 \times 2$ complex Monge-Ampère residual $|\det(g + H_\phi) / \det(g) - e^F|$
    /// for diagonal background metric $g = \operatorname{diag}(g_1, g_2)$ and complex Hessian $H_\phi = [[h_{00}, h_{01}], [h_{10}, h_{11}]]$.
    pub fn monge_ampere_2d_residual(
        g_diag: [f64; 2],
        hessian_phi: [f64; 4],
        ricci_potential_f: f64,
    ) -> Result<f64> {
        let [g0, g1] = g_diag;
        if g0 <= 0.0 || g1 <= 0.0 {
            return Err(Error::Msg(
                "Background Kahler metric must be positive definite".to_string(),
            ));
        }
        let [h00, h01, h10, h11] = hessian_phi;
        let det_deformed = (g0 + h00) * (g1 + h11) - h01 * h10;
        let ratio = det_deformed / (g0 * g1);
        Ok((ratio - ricci_potential_f.exp()).abs())
    }
}
