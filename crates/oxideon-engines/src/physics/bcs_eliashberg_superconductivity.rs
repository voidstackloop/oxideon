//! Bardeen-Cooper-Schrieffer (BCS) Theory of Superconductivity, Cooper Pair Gap $\Delta_0$ & Universal Ratio $\frac{2\Delta_0}{k_B T_c} \approx 3.528$ (BCS, 1957).
//!
//! ### 1. Weak-Coupling BCS Superconducting Gap & Critical Temperature
//! With attractive phonon-mediated pairing interaction $\lambda = N(0) V > 0$ and Debye cutoff energy $\hbar \omega_D$:
//! $$\Delta_0 = 2 \hbar \omega_D \exp\left(-\frac{1}{N(0) V}\right), \qquad \frac{2 \Delta_0}{k_B T_c} = \frac{2 \pi}{e^{\gamma_E}} \approx 3.52775$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Bardeen-Cooper-Schrieffer (BCS) Superconducting Gap & Critical Temperature Engine.
#[derive(Debug, Clone, Copy)]
pub struct BcsGapEquationEngine {
    /// Debye phonon cutoff energy $\hbar \omega_D > 0$.
    pub debye_energy: f64,
    /// Dimensionless electron-phonon coupling $\lambda = N(0) V > 0$.
    pub coupling_lambda: f64,
}

impl BcsGapEquationEngine {
    /// Creates a new BCS Superconductivity engine.
    pub fn new(debye_energy: f64, coupling_lambda: f64) -> Result<Self> {
        if debye_energy <= 0.0 || coupling_lambda <= 0.0 {
            return Err(Error::Msg(
                "debye_energy and coupling_lambda must be > 0".to_string(),
            ));
        }
        Ok(Self {
            debye_energy,
            coupling_lambda,
        })
    }

    /// Computes the zero-temperature BCS superconducting gap $\Delta_0$, critical temperature $k_B T_c$, and universal BCS ratio $2\Delta_0 / (k_B T_c) \approx 3.52775$.
    pub fn gap_tc_and_universal_ratio(&self) -> (f64, f64, f64) {
        let euler_gamma = 0.577_215_664_901_532_9_f64;
        let ratio = 2.0 * PI / euler_gamma.exp();
        let delta_0 = 2.0 * self.debye_energy * (-1.0 / self.coupling_lambda).exp();
        let tc = 2.0 * delta_0 / ratio;
        (delta_0, tc, ratio)
    }
}
