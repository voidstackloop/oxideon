//! Cavity Optomechanics, Radiation-Pressure Dynamical Backaction & Quantum Cooperativity (Aspelmeyer, Kippenberg & Marquardt, 2014).
//!
//! ### 1. Optomechanical Cooperativity $C$ & Sideband Cooling Final Occupancy
//! For linearized optomechanical coupling $g = g_0 \sqrt{\bar{n}_{\mathrm{cav}}}$, cavity decay rate $\kappa$, and mechanical damping $\gamma_m$:
//! $$C = \frac{4 g^2}{\kappa \gamma_m}, \qquad C_{\mathrm{qu}} = \frac{C}{n_{\mathrm{th}}}, \qquad \bar{n}_f \approx \frac{n_{\mathrm{th}}}{1 + C} + \left(\frac{\kappa}{4 \Omega_m}\right)^2$$

use crate::error::{Error, Result};

/// Cavity Optomechanics Radiation-Pressure Backaction & Ground-State Cooling Engine.
#[derive(Debug, Clone, Copy)]
pub struct CavityOptomechanicsBackaction {
    /// Cavity energy decay rate $\kappa > 0$.
    pub kappa: f64,
    /// Mechanical damping rate $\gamma_m > 0$.
    pub gamma_m: f64,
    /// Mechanical resonance frequency $\Omega_m > 0$.
    pub omega_m: f64,
}

impl Default for CavityOptomechanicsBackaction {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            gamma_m: 1e-4,
            omega_m: 10.0,
        }
    }
}

impl CavityOptomechanicsBackaction {
    /// Creates a new Cavity Optomechanics engine.
    pub fn new(kappa: f64, gamma_m: f64, omega_m: f64) -> Result<Self> {
        if kappa <= 0.0 || gamma_m <= 0.0 || omega_m <= 0.0 {
            return Err(Error::Msg(
                "kappa, gamma_m, and omega_m must be > 0".to_string(),
            ));
        }
        Ok(Self {
            kappa,
            gamma_m,
            omega_m,
        })
    }

    /// Evaluates the multiphoton cooperativity $C = 4 g^2 / (\kappa \gamma_m)$, resolved-sideband final phonon occupancy $\bar{n}_f$, and ground-state cooling check ($\bar{n}_f < 1$).
    pub fn cooperativity_and_final_phonons(
        &self,
        enhanced_coupling_g: f64,
        thermal_bath_nth: f64,
    ) -> Result<(f64, f64, bool)> {
        if enhanced_coupling_g <= 0.0 || thermal_bath_nth < 0.0 {
            return Err(Error::Msg(
                "enhanced_coupling_g > 0 and thermal_bath_nth >= 0 required".to_string(),
            ));
        }
        let coop = 4.0 * enhanced_coupling_g.powi(2) / (self.kappa * self.gamma_m);
        let backaction_limit = (self.kappa / (4.0 * self.omega_m)).powi(2);
        let n_final = thermal_bath_nth / (1.0 + coop) + backaction_limit;
        Ok((coop, n_final, n_final < 1.0))
    }
}
