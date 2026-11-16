//! MSW (Mikheyev-Smirnov-Wolfenstein) Matter-Enhanced Neutrino Oscillation & Resonance Mixing (Wolfenstein, 1978; Mikheyev & Smirnov, 1985).
//!
//! ### 1. Effective Matter Mixing Angle $\theta_m$
//! In matter with Wolfenstein potential $A = 2\sqrt{2} G_F N_e E$ relative to $\Delta m^2$:
//! $$\sin^2(2\theta_m) = \frac{\sin^2(2\theta_0)}{\sin^2(2\theta_0) + (\cos(2\theta_0) - A / \Delta m^2)^2}$$
//! Maximal resonance ($\sin^2(2\theta_m) = 1$) occurs when $A / \Delta m^2 = \cos(2\theta_0)$.

use crate::error::{Error, Result};

/// MSW Matter-Enhanced Neutrino Oscillation & Resonance Engine.
#[derive(Debug, Clone, Copy)]
pub struct MswNeutrinoOscillationEngine {
    /// Vacuum mixing angle $\theta_0$ in radians.
    pub theta_0: f64,
    /// Mass-squared splitting $\Delta m^2 > 0$ ($\mathrm{eV}^2$).
    pub delta_m_sq: f64,
}

impl Default for MswNeutrinoOscillationEngine {
    fn default() -> Self {
        Self {
            theta_0: 0.58,
            delta_m_sq: 7.5e-5,
        }
    }
}

impl MswNeutrinoOscillationEngine {
    /// Creates a new MSW Neutrino Oscillation engine.
    pub fn new(theta_0: f64, delta_m_sq: f64) -> Result<Self> {
        if delta_m_sq <= 0.0 || !delta_m_sq.is_finite() {
            return Err(Error::Msg("delta_m_sq must be > 0".to_string()));
        }
        Ok(Self {
            theta_0,
            delta_m_sq,
        })
    }

    /// Evaluates the effective matter mixing amplitude $\sin^2(2\theta_m)$ and effective splitting $\Delta m_m^2$ for matter potential $A$.
    pub fn matter_mixing_and_splitting(&self, wolfenstein_a: f64) -> (f64, f64) {
        let s2 = (2.0 * self.theta_0).sin().powi(2);
        let c = (2.0 * self.theta_0).cos();
        let ratio = wolfenstein_a / self.delta_m_sq;
        let denom = s2 + (c - ratio).powi(2);
        let sin2_2theta_m = if denom > 0.0 { s2 / denom } else { 1.0 };
        let dm_m_sq = self.delta_m_sq * denom.sqrt();
        (sin2_2theta_m, dm_m_sq)
    }
}
