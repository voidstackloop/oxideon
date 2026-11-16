//! Active Nematics, $+1/2$ Disclination Self-Propulsion & Active Lengthscale $\ell_a = \sqrt{K / |\zeta|}$ (Simha & Ramaswamy, 2002; Giomi et al., 2013).
//!
//! ### 1. Active Stress $\sigma^{\mathrm{act}}_{ij} = -\zeta Q_{ij}$ & Defect Vortex Scale
//! Balancing Frank elastic constant $K$ against extensile/contractile activity $\zeta$ defines the active nematic lengthscale
//! $\ell_a = \sqrt{K / |\zeta|}$, while a comet-like $+1/2$ topological disclination self-propels at speed $v_{+1/2} \propto |\zeta| \ell_a / \eta$.

use crate::error::{Error, Result};

/// Active Nematic Turbulence & $+1/2$ Topological Defect Propulsion Engine.
#[derive(Debug, Clone, Copy)]
pub struct ActiveNematicDefectTurbulence {
    /// Frank nematic elastic constant $K > 0$.
    pub frank_elastic_k: f64,
    /// Shear viscosity $\eta > 0$.
    pub viscosity_eta: f64,
}

impl Default for ActiveNematicDefectTurbulence {
    fn default() -> Self {
        Self {
            frank_elastic_k: 4.0,
            viscosity_eta: 2.0,
        }
    }
}

impl ActiveNematicDefectTurbulence {
    /// Creates a new Active Nematic Defect Turbulence engine.
    pub fn new(frank_elastic_k: f64, viscosity_eta: f64) -> Result<Self> {
        if frank_elastic_k <= 0.0 || viscosity_eta <= 0.0 {
            return Err(Error::Msg(
                "frank_elastic_k and viscosity_eta must be > 0".to_string(),
            ));
        }
        Ok(Self {
            frank_elastic_k,
            viscosity_eta,
        })
    }

    /// Evaluates the active lengthscale $\ell_a = \sqrt{K / |\zeta|}$ and $+1/2$ defect self-propulsion speed $v_{+1/2} = |\zeta| \ell_a / \eta$ for activity $\zeta \ne 0$.
    pub fn active_length_and_defect_velocity(&self, activity_zeta: f64) -> Result<(f64, f64)> {
        if activity_zeta == 0.0 || !activity_zeta.is_finite() {
            return Err(Error::Msg("activity_zeta must be non-zero".to_string()));
        }
        let z_abs = activity_zeta.abs();
        let l_a = (self.frank_elastic_k / z_abs).sqrt();
        let v_def = z_abs * l_a / self.viscosity_eta;
        Ok((l_a, v_def))
    }
}
