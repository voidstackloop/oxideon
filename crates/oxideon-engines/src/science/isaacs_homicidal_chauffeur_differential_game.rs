//! Isaacs Zero-Sum Differential Games & Homicidal Chauffeur / Apollonius Capture Barrier (Isaacs, 1965).
//!
//! ### 1. Isaacs Minimax Hamiltonian $H(x, \nabla V) = \min_u \max_d \nabla V \cdot f(x, u, d) = 0$ & Apollonius Capture Circle
//! In planar pursuit-evasion with pursuer speed $v_P > v_E$ and speed ratio $\gamma = v_E / v_P \in (0, 1)$, the locus of equal arrival times is the Apollonius circle of radius:
//! $$R_{\mathrm{Apollonius}} = \frac{\gamma}{1 - \gamma^2} \|x_P - x_E\|$$

use crate::error::{Error, Result};

/// Isaacs Differential Pursuit-Evasion Game & Apollonius Barrier Engine.
#[derive(Debug, Clone, Copy)]
pub struct IsaacsDifferentialPursuitGame {
    /// Pursuer speed $v_P > 0$.
    pub pursuer_speed: f64,
    /// Evader speed $0 < v_E < v_P$.
    pub evader_speed: f64,
}

impl Default for IsaacsDifferentialPursuitGame {
    fn default() -> Self {
        Self {
            pursuer_speed: 5.0,
            evader_speed: 3.0,
        }
    }
}

impl IsaacsDifferentialPursuitGame {
    /// Creates a new Isaacs Differential Pursuit Game engine.
    pub fn new(pursuer_speed: f64, evader_speed: f64) -> Result<Self> {
        if pursuer_speed <= evader_speed || evader_speed <= 0.0 {
            return Err(Error::Msg(
                "0 < evader_speed < pursuer_speed required".to_string(),
            ));
        }
        Ok(Self {
            pursuer_speed,
            evader_speed,
        })
    }

    /// Evaluates the Apollonius capture circle radius $R_A$ and pure-pursuit capture time $d_0 / (v_P - v_E)$.
    pub fn apollonius_radius_and_capture_time(
        &self,
        initial_distance_d0: f64,
    ) -> Result<(f64, f64)> {
        if initial_distance_d0 <= 0.0 || !initial_distance_d0.is_finite() {
            return Err(Error::Msg("initial_distance_d0 must be > 0".to_string()));
        }
        let gamma = self.evader_speed / self.pursuer_speed;
        let r_ap = (gamma / (1.0 - gamma * gamma)) * initial_distance_d0;
        let t_cap = initial_distance_d0 / (self.pursuer_speed - self.evader_speed);
        Ok((r_ap, t_cap))
    }
}
