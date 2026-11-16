//! Pastawski-Yoshida-Harlow-Preskill (HaPPY) Pentagonal Holographic Quantum Error-Correcting Code & Entanglement Wedge Reconstruction (Pastawski, Yoshida, Harlow & Preskill, 2015).
//!
//! ### 1. Hyperbolic $\{5, 4\}$ Pentagonal Tessellation Growth & Ryu-Takayanagi Code Rate
//! In the HaPPY $\{5, 4\}$ hyperbolic tensor network of radial layer depth $L$, each 6-leg perfect tensor encodes 1 bulk logical qubit into boundary physical legs with asymptotic radial boundary growth $\lambda_{\mathrm{hyp}} = 2 + \sqrt{3} \approx 3.73205$.

use crate::error::{Error, Result};

/// HaPPY Holographic Pentagonal Tensor Code & Bulk Reconstruction Engine.
#[derive(Debug, Clone, Copy)]
pub struct HappyHolographicCodeBulkWedge {
    /// Hyperbolic radial layer depth $L \ge 1$.
    pub radial_layers_l: u32,
}

impl Default for HappyHolographicCodeBulkWedge {
    fn default() -> Self {
        Self { radial_layers_l: 3 }
    }
}

impl HappyHolographicCodeBulkWedge {
    /// Creates a new HaPPY Holographic Code engine.
    pub fn new(radial_layers_l: u32) -> Result<Self> {
        if radial_layers_l == 0 || radial_layers_l > 25 {
            return Err(Error::Msg("radial_layers_l must be in 1..=25".to_string()));
        }
        Ok(Self { radial_layers_l })
    }

    /// Evaluates the asymptotic $\{5, 4\}$ hyperbolic growth ratio $\lambda = 2 + \sqrt{3}$ and boundary-to-bulk perimeter scaling $\lambda^L$.
    pub fn hyperbolic_growth_ratio_and_boundary_scale(&self) -> (f64, f64) {
        let lam = 2.0 + 3.0f64.sqrt();
        (lam, lam.powi(self.radial_layers_l as i32))
    }
}
