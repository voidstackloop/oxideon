//! Willems Dissipativity Inequality $\dot{S}(x) \le w(u, y)$ & Kalman-Yakubovich-Popov (KYP) Passivity Lemma (Yakubovich, 1962; Kalman, 1963; Willems, 1972).
//!
//! ### 1. Storage Function Dissipation Inequality & Input-Feedforward Output-Feedback Passivity Indices $(\nu, \rho)$
//! A system with input $u$ and output $y$ is $(\nu, \rho)$-excess passive with storage function $S(x) \ge 0$ if along trajectories:
//! $$\dot{S}(x) \le u^\top y - \nu \|u\|^2 - \rho \|y\|^2$$

use crate::error::{Error, Result};

/// Willems Dissipativity & KYP Passivity Index Engine.
#[derive(Debug, Clone, Copy)]
pub struct WillemsDissipativityKypEngine {
    /// Input-feedforward passivity index $\nu \in \mathbb{R}$.
    pub ifp_nu: f64,
    /// Output-feedback passivity index $\rho \in \mathbb{R}$.
    pub ofp_rho: f64,
}

impl Default for WillemsDissipativityKypEngine {
    fn default() -> Self {
        Self {
            ifp_nu: 0.2,
            ofp_rho: 0.3,
        }
    }
}

impl WillemsDissipativityKypEngine {
    /// Creates a new Willems Dissipativity & KYP engine.
    pub fn new(ifp_nu: f64, ofp_rho: f64) -> Result<Self> {
        if !ifp_nu.is_finite() || !ofp_rho.is_finite() {
            return Err(Error::Msg("ifp_nu and ofp_rho must be finite".to_string()));
        }
        Ok(Self { ifp_nu, ofp_rho })
    }

    /// Evaluates the Willems supply rate $w(u, y) = u y - \nu u^2 - \rho y^2$ and checks whether the system is strictly passive ($\nu > 0$ and $\rho > 0$).
    pub fn supply_rate_and_strict_passivity(&self, input_u: f64, output_y: f64) -> (f64, bool) {
        let supply = input_u * output_y
            - self.ifp_nu * input_u * input_u
            - self.ofp_rho * output_y * output_y;
        (supply, self.ifp_nu > 0.0 && self.ofp_rho > 0.0)
    }
}
