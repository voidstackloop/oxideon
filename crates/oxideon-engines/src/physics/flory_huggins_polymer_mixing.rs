//! Flory-Huggins Polymer Solution Thermodynamics & Spinodal Critical Point $(\phi_c, \chi_c)$ (Flory, 1942; Huggins, 1942).
//!
//! ### 1. Free Energy per Lattice Site & Critical Interaction Parameter $\chi_c$
//! For a polymer of degree $N \ge 1$ in solvent ($N_s = 1$) at volume fraction $\phi \in (0, 1)$ with Flory interaction parameter $\chi$:
//! $$\frac{\Delta f_{\mathrm{mix}}}{k_B T} = \frac{\phi}{N} \ln \phi + (1 - \phi) \ln(1 - \phi) + \chi \phi (1 - \phi), \qquad \chi_c = \frac{1}{2}\!\left(1 + \frac{1}{\sqrt{N}}\right)^2$$

use crate::error::{Error, Result};

/// Flory-Huggins Polymer Mixing & Liquid-Liquid Phase Separation (LLPS) Engine.
#[derive(Debug, Clone, Copy)]
pub struct FloryHugginsSpinodalMixing {
    /// Polymer chain degree of polymerization $N \ge 1$.
    pub degree_n: usize,
}

impl Default for FloryHugginsSpinodalMixing {
    fn default() -> Self {
        Self { degree_n: 100 }
    }
}

impl FloryHugginsSpinodalMixing {
    /// Creates a new Flory-Huggins Spinodal Mixing engine.
    pub fn new(degree_n: usize) -> Result<Self> {
        if degree_n == 0 {
            return Err(Error::Msg("degree_n must be >= 1".to_string()));
        }
        Ok(Self { degree_n })
    }

    /// Evaluates the critical volume fraction $\phi_c = 1 / (1 + \sqrt{N})$, critical parameter $\chi_c = \frac{1}{2}(1 + 1/\sqrt{N})^2$, and checks spinodal instability ($\partial^2 f / \partial \phi^2 < 0$).
    pub fn critical_point_and_spinodal_instability(
        &self,
        phi: f64,
        chi: f64,
    ) -> Result<(f64, f64, bool)> {
        if !(0.0..1.0).contains(&phi) || phi == 0.0 {
            return Err(Error::Msg("phi must be in (0, 1)".to_string()));
        }
        let n = self.degree_n as f64;
        let sqrt_n = n.sqrt();
        let phi_c = 1.0 / (1.0 + sqrt_n);
        let chi_c = 0.5 * (1.0 + 1.0 / sqrt_n).powi(2);
        let d2f = 1.0 / (n * phi) + 1.0 / (1.0 - phi) - 2.0 * chi;
        Ok((phi_c, chi_c, d2f < 0.0))
    }
}
