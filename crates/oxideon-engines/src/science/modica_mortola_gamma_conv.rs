//! De Giorgi $\Gamma$-Convergence & Modica-Mortola Phase-Field Interfacial Perimeter Theorem (De Giorgi, 1975; Modica & Mortola, 1977).
//!
//! ### 1. Modica-Mortola $\Gamma$-Limit to Caccioppoli Perimeter
//! As interface width $\varepsilon \to 0^+$, the Ginzburg-Landau / Cahn-Hilliard functional $F_\varepsilon(u) = \int \left(\frac{\varepsilon}{2} |\nabla u|^2 + \frac{1}{\varepsilon} W(u)\right) dx$
//! with double-well $W(u) = \frac{1}{4}(1 - u^2)^2$ $\Gamma$-converges in $L^1$ to $\sigma \, \mathrm{Per}(\{u = 1\})$, where:
//! $$\sigma = \int_{-1}^{1} \sqrt{2 W(s)} \, ds = \frac{2\sqrt{2}}{3}, \qquad u_\varepsilon(x) = \tanh\!\left(\frac{x}{\varepsilon}\right)$$

use crate::error::{Error, Result};

/// De Giorgi $\Gamma$-Convergence & Modica-Mortola Phase-Field Engine.
#[derive(Debug, Clone, Copy)]
pub struct ModicaMortolaGammaConvergence {
    /// Diffuse interface thickness $\varepsilon > 0$.
    pub epsilon: f64,
}

impl Default for ModicaMortolaGammaConvergence {
    fn default() -> Self {
        Self { epsilon: 0.05 }
    }
}

impl ModicaMortolaGammaConvergence {
    /// Creates a new Modica-Mortola $\Gamma$-Convergence engine.
    pub fn new(epsilon: f64) -> Result<Self> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(Error::Msg(
                "Interface width epsilon must be > 0".to_string(),
            ));
        }
        Ok(Self { epsilon })
    }

    /// Evaluates the exact Modica-Mortola surface tension constant $\sigma = \frac{2\sqrt{2}}{3}$ and the optimal 1D transition profile $u_\varepsilon(x) = \tanh(x / (\sqrt{2} \varepsilon))$.
    pub fn surface_tension_and_optimal_profile(&self, x: f64) -> (f64, f64) {
        let sigma = 2.0 * 2.0_f64.sqrt() / 3.0;
        let u_x = (x / (2.0_f64.sqrt() * self.epsilon)).tanh();
        (sigma, u_x)
    }
}
