//! De Gennes–Doi–Edwards Polymer Reptation Tube Theory & Entangled Melt Scaling (de Gennes, 1971; Doi & Edwards, 1986).
//!
//! ### 1. Reptation Disengagement Time $\tau_d \propto N^3$ & Center-of-Mass Diffusivity $D_{\mathrm{rep}} \propto N^{-2}$
//! For a polymer chain of $N > N_e$ segments (with entanglement length $N_e$ and entanglement Rouse time $\tau_e$):
//! $$\tau_d = 3 \tau_e \left(\frac{N}{N_e}\right)^3, \qquad D_{\mathrm{rep}} = D_e \left(\frac{N_e}{N}\right)^2$$

use crate::error::{Error, Result};

/// De Gennes–Doi–Edwards Polymer Reptation Tube Engine.
#[derive(Debug, Clone, Copy)]
pub struct DeGennesPolymerReptationTube {
    /// Entanglement degree of polymerization $N_e > 0$.
    pub entanglement_ne: f64,
    /// Entanglement Rouse relaxation time $\tau_e > 0$.
    pub tau_e: f64,
}

impl Default for DeGennesPolymerReptationTube {
    fn default() -> Self {
        Self {
            entanglement_ne: 50.0,
            tau_e: 1.0,
        }
    }
}

impl DeGennesPolymerReptationTube {
    /// Creates a new De Gennes Polymer Reptation Tube engine.
    pub fn new(entanglement_ne: f64, tau_e: f64) -> Result<Self> {
        if entanglement_ne <= 0.0 || tau_e <= 0.0 {
            return Err(Error::Msg(
                "entanglement_ne and tau_e must be > 0".to_string(),
            ));
        }
        Ok(Self {
            entanglement_ne,
            tau_e,
        })
    }

    /// Evaluates the number of entanglements $Z = N / N_e$, tube disengagement time $\tau_d = 3 \tau_e Z^3$, and zero-shear viscosity ratio $\eta_0 / \eta_e = Z^{3.4}$ (including contour-length fluctuations).
    pub fn entanglements_disengagement_and_viscosity(
        &self,
        chain_length_n: f64,
    ) -> Result<(f64, f64, f64)> {
        if chain_length_n <= self.entanglement_ne {
            return Err(Error::Msg(
                "chain_length_n must exceed entanglement_ne".to_string(),
            ));
        }
        let z = chain_length_n / self.entanglement_ne;
        let tau_d = 3.0 * self.tau_e * z.powi(3);
        let eta_ratio = z.powf(3.4);
        Ok((z, tau_d, eta_ratio))
    }
}
