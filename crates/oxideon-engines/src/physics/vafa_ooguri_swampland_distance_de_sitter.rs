//! Vafa-Ooguri Swampland Distance Conjecture $m(\phi) \sim m_0 e^{-\alpha \Delta \phi}$ & de Sitter Gradient Bound (Vafa, 2005; Ooguri & Vafa, 2007; Obied et al., 2018).
//!
//! ### 1. Exponential Tower Mass Scale & de Sitter Potential Slope $|\nabla V| / V \ge c_{\mathrm{dS}}$
//! At large geodesic moduli displacement $\Delta \phi$ (in Planck units $M_{\mathrm{Pl}} = 1$), an infinite tower of states becomes exponentially light:
//! $$m(\Delta \phi) = m_0 \exp(-\alpha \Delta \phi), \qquad \alpha \ge \frac{1}{\sqrt{d - 2}}$$

use crate::error::{Error, Result};

/// Vafa-Ooguri Swampland Distance & de Sitter Conjecture Engine.
#[derive(Debug, Clone, Copy)]
pub struct SwamplandDistanceDeSitterBounds {
    /// Spacetime dimension $d \ge 4$.
    pub spacetime_dim_d: usize,
}

impl Default for SwamplandDistanceDeSitterBounds {
    fn default() -> Self {
        Self { spacetime_dim_d: 4 }
    }
}

impl SwamplandDistanceDeSitterBounds {
    /// Creates a new Swampland Distance & de Sitter Bounds engine.
    pub fn new(spacetime_dim_d: usize) -> Result<Self> {
        if spacetime_dim_d < 4 {
            return Err(Error::Msg("spacetime_dim_d must be >= 4".to_string()));
        }
        Ok(Self { spacetime_dim_d })
    }

    /// Evaluates the Emergent String Conjecture minimum exponent $\alpha_{\min} = 1 / \sqrt{d - 2}$ and tower mass ratio $m(\Delta \phi)/m_0 = \exp(-\alpha_{\min} \Delta \phi)$.
    pub fn min_decay_rate_and_tower_mass_ratio(
        &self,
        geodesic_distance_dphi: f64,
    ) -> Result<(f64, f64)> {
        if geodesic_distance_dphi < 0.0 || !geodesic_distance_dphi.is_finite() {
            return Err(Error::Msg(
                "geodesic_distance_dphi must be >= 0".to_string(),
            ));
        }
        let alpha_min = 1.0 / ((self.spacetime_dim_d - 2) as f64).sqrt();
        let ratio = (-alpha_min * geodesic_distance_dphi).exp();
        Ok((alpha_min, ratio))
    }
}
