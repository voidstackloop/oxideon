//! Unruh-DeWitt Detector Thermalization, Rindler Horizon Acceleration Temperature & Bogoliubov Mixing (Unruh, 1976).
//!
//! ### 1. Unruh Temperature & Bogoliubov Planck Occupation
//! A uniformly accelerated observer with proper acceleration $a$ in Minkowski vacuum perceives a thermal bath at temperature
//! $T_U = \frac{a}{2 \pi}$ (in natural units $\hbar = c = k_B = 1$) with Bogoliubov squeezing coefficient $|\beta_\omega|^2 = (e^{2\pi \omega / a} - 1)^{-1}$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Unruh-DeWitt Detector & Rindler Bogoliubov Thermalization Engine.
#[derive(Debug, Clone, Copy)]
pub struct UnruhRindlerBogoliubovEngine {
    /// Proper acceleration $a > 0$ in natural units.
    pub proper_acceleration_a: f64,
}

impl Default for UnruhRindlerBogoliubovEngine {
    fn default() -> Self {
        Self {
            proper_acceleration_a: 2.0 * PI,
        }
    }
}

impl UnruhRindlerBogoliubovEngine {
    /// Creates a new Unruh-Rindler Bogoliubov engine.
    pub fn new(proper_acceleration_a: f64) -> Result<Self> {
        if proper_acceleration_a <= 0.0 || !proper_acceleration_a.is_finite() {
            return Err(Error::Msg("proper_acceleration_a must be > 0".to_string()));
        }
        Ok(Self {
            proper_acceleration_a,
        })
    }

    /// Evaluates the Unruh temperature $T_U = a / (2\pi)$ and Bogoliubov occupation number $|\beta_\omega|^2 = 1 / (e^{\omega / T_U} - 1)$.
    pub fn unruh_temperature_and_bogoliubov_n(&self, mode_omega: f64) -> Result<(f64, f64)> {
        if mode_omega <= 0.0 {
            return Err(Error::Msg("mode_omega must be > 0".to_string()));
        }
        let t_u = self.proper_acceleration_a / (2.0 * PI);
        let n_omega = 1.0 / ((mode_omega / t_u).exp() - 1.0);
        Ok((t_u, n_omega))
    }
}
