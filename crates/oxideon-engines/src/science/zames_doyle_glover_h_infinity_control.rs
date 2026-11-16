//! Zames Small-Gain Theorem & Doyle-Glover-Khargonekar-Francis Two-Riccati $H_\infty$ Robust Control (Zames, 1966; Doyle, Glover, Khargonekar & Francis, 1989).
//!
//! ### 1. Small-Gain Robust Stability Condition $\|\Delta\|_\infty \|M\|_\infty < 1$ & $H_\infty$ Game Riccati Sign-Indefinite Term
//! In $H_\infty$ suboptimal control ($\|T_{zw}\|_\infty < \gamma$), the state-feedback game Algebraic Riccati Equation modifies the LQR quadratic term by disturbance attenuation $-\gamma^{-2} B_1 B_1^\top$:
//! $$A^\top X + X A + C_1^\top C_1 - X \left(B_2 B_2^\top - \gamma^{-2} B_1 B_1^\top\right) X = 0$$

use crate::error::{Error, Result};

/// Zames Small-Gain & DGKF $H_\infty$ Robust Control Engine.
#[derive(Debug, Clone, Copy)]
pub struct ZamesDoyleHInfinityControl {
    /// Nominal closed-loop $H_\infty$ norm $\|M\|_\infty > 0$.
    pub nominal_h_inf_norm: f64,
}

impl Default for ZamesDoyleHInfinityControl {
    fn default() -> Self {
        Self {
            nominal_h_inf_norm: 0.5,
        }
    }
}

impl ZamesDoyleHInfinityControl {
    /// Creates a new Zames-Doyle $H_\infty$ Control engine.
    pub fn new(nominal_h_inf_norm: f64) -> Result<Self> {
        if nominal_h_inf_norm <= 0.0 || !nominal_h_inf_norm.is_finite() {
            return Err(Error::Msg("nominal_h_inf_norm must be > 0".to_string()));
        }
        Ok(Self { nominal_h_inf_norm })
    }

    /// Evaluates the small-gain stability margin $1 / \|M\|_\infty$ and checks whether uncertainty norm $\|\Delta\|_\infty$ preserves robust stability.
    pub fn max_allowable_uncertainty_and_stability(&self, delta_norm: f64) -> Result<(f64, bool)> {
        if delta_norm < 0.0 || !delta_norm.is_finite() {
            return Err(Error::Msg("delta_norm must be >= 0".to_string()));
        }
        let max_delta = 1.0 / self.nominal_h_inf_norm;
        Ok((max_delta, delta_norm * self.nominal_h_inf_norm < 1.0))
    }
}
