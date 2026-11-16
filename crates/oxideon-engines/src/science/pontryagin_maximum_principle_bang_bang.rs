//! Pontryagin Maximum Principle (PMP) & Bushaw-Pontryagin Minimum-Time Double-Integrator Bang-Bang Control (Pontryagin, Boltyanskii, Gamkrelidze & Mishchenko, 1962).
//!
//! ### 1. Minimum-Time Switching Curve $x_1 = -\frac{\operatorname{sgn}(x_2)}{2 u_{\max}} x_2^2$ & Optimal Time $T^*(x_0, v_0)$
//! For $\ddot{x} = u$ with $|u(t)| \le u_{\max}$, driving $(x_0, 0)$ to $(0, 0)$ in minimum time uses bang-bang control $u^*(t) \in \{\pm u_{\max}\}$ with one switch at $t_s = \sqrt{|x_0| / u_{\max}}$ and total minimum time:
//! $$T^*(x_0, 0) = 2 \sqrt{\frac{|x_0|}{u_{\max}}}$$

use crate::error::{Error, Result};

/// Pontryagin Maximum Principle & Bang-Bang Minimum-Time Control Engine.
#[derive(Debug, Clone, Copy)]
pub struct PontryaginMinimumTimeBangBang {
    /// Control saturation bound $u_{\max} > 0$.
    pub u_max: f64,
}

impl Default for PontryaginMinimumTimeBangBang {
    fn default() -> Self {
        Self { u_max: 2.0 }
    }
}

impl PontryaginMinimumTimeBangBang {
    /// Creates a new Pontryagin Bang-Bang Minimum-Time engine.
    pub fn new(u_max: f64) -> Result<Self> {
        if u_max <= 0.0 || !u_max.is_finite() {
            return Err(Error::Msg("u_max must be > 0".to_string()));
        }
        Ok(Self { u_max })
    }

    /// Evaluates the rest-to-rest switching time $t_s = \sqrt{|x_0| / u_{\max}}$ and total minimum time $T^* = 2 t_s$.
    pub fn switching_time_and_minimum_total_time(&self, initial_pos_x0: f64) -> Result<(f64, f64)> {
        if !initial_pos_x0.is_finite() {
            return Err(Error::Msg("initial_pos_x0 must be finite".to_string()));
        }
        let ts = (initial_pos_x0.abs() / self.u_max).sqrt();
        Ok((ts, 2.0 * ts))
    }
}
