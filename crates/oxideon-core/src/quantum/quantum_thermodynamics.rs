//! Quantum Otto Heat Engine, Curzon-Ahlborn Efficiency & Spohn Irreversible Entropy Production.
//!
//! ### 1. 4-Stroke Quantum Otto Cycle of a Two-Level Qubit / Harmonic Oscillator
//! Operating between hot bath inverse temperature $\beta_H$ (at frequency $\omega_H$) and cold bath $\beta_C > \beta_H$
//! (at frequency $\omega_C < \omega_H$) with adiabatic compression ratio $\eta_O = 1 - \frac{\omega_C}{\omega_H}$,
//! positive work extraction occurs in the regime $\frac{\beta_H}{\beta_C} < \frac{\omega_C}{\omega_H} < 1$,
//! strictly bounded by the Carnot efficiency $\eta_O < 1 - \frac{T_C}{T_H} = \eta_{\text{Carnot}}$.

use crate::error::{Error, Result};

/// Four-Stroke Quantum Otto Thermodynamic Cycle Engine.
#[derive(Debug, Clone, Copy)]
pub struct QuantumOttoEngine {
    /// Hot frequency $\omega_H > 0$.
    pub omega_h: f64,
    /// Cold frequency $\omega_C \in (0, \omega_H)$.
    pub omega_c: f64,
    /// Hot bath temperature $T_H > 0$.
    pub temp_h: f64,
    /// Cold bath temperature $T_C \in (0, T_H)$.
    pub temp_c: f64,
}

impl QuantumOttoEngine {
    /// Constructs a new Quantum Otto Engine and validates thermodynamic parameters.
    pub fn new(omega_h: f64, omega_c: f64, temp_h: f64, temp_c: f64) -> Result<Self> {
        if omega_c <= 0.0 || omega_h <= omega_c || temp_c <= 0.0 || temp_h <= temp_c {
            return Err(Error::Msg(
                "Requires 0 < omega_c < omega_h and 0 < temp_c < temp_h".to_string(),
            ));
        }
        Ok(Self {
            omega_h,
            omega_c,
            temp_h,
            temp_c,
        })
    }

    /// Computes the Quantum Otto efficiency $\eta_{\text{Otto}} = 1 - \omega_C / \omega_H$ and Carnot bound $\eta_{\text{Carnot}} = 1 - T_C / T_H$.
    pub fn otto_and_carnot_efficiencies(&self) -> (f64, f64) {
        (
            1.0 - self.omega_c / self.omega_h,
            1.0 - self.temp_c / self.temp_h,
        )
    }

    /// Computes net work extracted per cycle $W = \frac{1}{2}(\omega_H - \omega_C) \left(\tanh(\frac{\omega_C}{2 T_C}) - \tanh(\frac{\omega_H}{2 T_H})\right)$.
    pub fn work_per_cycle(&self) -> f64 {
        let pop_diff =
            (0.5 * self.omega_c / self.temp_c).tanh() - (0.5 * self.omega_h / self.temp_h).tanh();
        0.5 * (self.omega_h - self.omega_c) * pop_diff
    }
}
