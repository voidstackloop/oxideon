//! Hohenberg-Kohn-Sham Density Functional Theory (DFT), Dirac LDA Exchange & Perdew-Burke-Ernzerhof (PBE) GGA (Kohn & Sham, 1965; Perdew, Burke & Ernzerhof, 1996).
//!
//! ### 1. Dirac-Slater Local Exchange $\varepsilon_x^{\mathrm{LDA}}(n)$ & PBE Enhancement Factor $F_x^{\mathrm{PBE}}(s)$
//! For uniform electron gas density $n > 0$ and dimensionless reduced density gradient $s = \frac{|\nabla n|}{2 (3\pi^2)^{1/3} n^{4/3}}$:
//! $$\varepsilon_x^{\mathrm{LDA}}(n) = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} n^{1/3}, \qquad F_x^{\mathrm{PBE}}(s) = 1 + \kappa - \frac{\kappa}{1 + \mu s^2 / \kappa}$$
//! where $\kappa = 0.804$ (Lieb-Oxford bound) and $\mu = 0.21951$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Kohn-Sham DFT Jacob's Ladder (LDA + PBE GGA) Exchange-Correlation Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct KohnShamJacobLadderEngine;

impl KohnShamJacobLadderEngine {
    /// Creates a new Kohn-Sham DFT engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the PBE GGA exchange energy density $e_x^{\mathrm{PBE}}(n, s) = n \, \varepsilon_x^{\mathrm{LDA}}(n) \, F_x^{\mathrm{PBE}}(s)$.
    pub fn pbe_exchange_energy_density(density_n: f64, reduced_gradient_s: f64) -> Result<f64> {
        if density_n <= 0.0 || reduced_gradient_s < 0.0 {
            return Err(Error::Msg(
                "density_n > 0 and reduced_gradient_s >= 0 required".to_string(),
            ));
        }
        let cx = -0.75 * (3.0 / PI).powf(1.0 / 3.0);
        let ex_lda = cx * density_n.powf(4.0 / 3.0);
        let kappa = 0.804_f64;
        let mu = 0.21951_f64;
        let fx_pbe =
            1.0 + kappa - kappa / (1.0 + mu * reduced_gradient_s * reduced_gradient_s / kappa);
        Ok(ex_lda * fx_pbe)
    }
}
