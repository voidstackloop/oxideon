//! Michaelis-Menten & Briggs-Haldane Quasi-Steady-State Enzyme Kinetics & Diffusion Perfection Limit (Michaelis & Menten, 1913; Briggs & Haldane, 1925).
//!
//! ### 1. Catalytic Efficiency $k_{\mathrm{cat}} / K_M$ & Competitive Inhibition Velocity
//! Under Briggs-Haldane steady-state $K_M = (k_{-1} + k_{\mathrm{cat}}) / k_1$ and competitive inhibitor $\[I\]$ with dissociation constant $K_I$:
//! $$v = \frac{k_{\mathrm{cat}} \[E\]_0 \[S\]}{K_M (1 + \[I\]/K_I) + \[S\]}$$

use crate::error::{Error, Result};

/// Briggs-Haldane Enzyme Kinetics & Catalytic Perfection Engine.
#[derive(Debug, Clone, Copy)]
pub struct BriggsHaldaneEnzymeKinetics {
    /// Turnover number $k_{\mathrm{cat}} > 0$ ($\mathrm{s^{-1}}$).
    pub k_cat: f64,
    /// Michaelis constant $K_M > 0$ (M).
    pub k_m: f64,
}

impl Default for BriggsHaldaneEnzymeKinetics {
    fn default() -> Self {
        Self {
            k_cat: 1000.0,
            k_m: 1.0e-5,
        }
    }
}

impl BriggsHaldaneEnzymeKinetics {
    /// Creates a new Briggs-Haldane Enzyme Kinetics engine.
    pub fn new(k_cat: f64, k_m: f64) -> Result<Self> {
        if k_cat <= 0.0 || k_m <= 0.0 {
            return Err(Error::Msg("k_cat and k_m must be > 0".to_string()));
        }
        Ok(Self { k_cat, k_m })
    }

    /// Evaluates catalytic specificity $k_{\mathrm{cat}} / K_M$ and reaction velocity $v / \[E\]_0$ under substrate $\[S\]$ and competitive inhibitor ratio $\[I\]/K_I \ge 0$.
    pub fn specificity_and_velocity(
        &self,
        substrate_s: f64,
        inhibitor_over_ki: f64,
    ) -> Result<(f64, f64)> {
        if substrate_s < 0.0 || inhibitor_over_ki < 0.0 {
            return Err(Error::Msg(
                "substrate_s and inhibitor_over_ki must be >= 0".to_string(),
            ));
        }
        let specificity = self.k_cat / self.k_m;
        let km_app = self.k_m * (1.0 + inhibitor_over_ki);
        let v_over_e0 = self.k_cat * substrate_s / (km_app + substrate_s);
        Ok((specificity, v_over_e0))
    }
}
