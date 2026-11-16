//! Continuous-Time Kalman-Bucy Filter, Algebraic Riccati Equation (ARE) & LQG Separation Principle (Kalman & Bucy, 1961).
//!
//! ### 1. Continuous-Time Differential & Algebraic Riccati Equation
//! For scalar/diagonal linear stochastic systems $dx_t = a x_t dt + \sqrt{q} dW_t$ with noisy observation $dy_t = c x_t dt + \sqrt{r} dV_t$,
//! the steady-state error variance $P_\infty > 0$ satisfies the **Algebraic Riccati Equation (ARE)**:
//! $$2 a P_\infty + q - \frac{c^2}{r} P_\infty^2 = 0 \quad \Longrightarrow \quad P_\infty = \frac{r}{c^2}\left(a + \sqrt{a^2 + \frac{c^2 q}{r}}\right)$$

use crate::error::{Error, Result};

/// Continuous-Time Kalman-Bucy Filter & Algebraic Riccati Engine.
#[derive(Debug, Clone, Copy)]
pub struct KalmanBucyRiccatiEngine {
    /// Drift coefficient $a$.
    pub drift_a: f64,
    /// Observation gain $c \ne 0$.
    pub obs_c: f64,
    /// Process noise variance $q > 0$.
    pub process_q: f64,
    /// Measurement noise variance $r > 0$.
    pub meas_r: f64,
}

impl KalmanBucyRiccatiEngine {
    /// Creates a new Kalman-Bucy Riccati engine.
    pub fn new(drift_a: f64, obs_c: f64, process_q: f64, meas_r: f64) -> Result<Self> {
        if obs_c.abs() < 1e-15 || process_q <= 0.0 || meas_r <= 0.0 {
            return Err(Error::Msg(
                "Non-zero obs_c and positive process_q, meas_r required".to_string(),
            ));
        }
        Ok(Self {
            drift_a,
            obs_c,
            process_q,
            meas_r,
        })
    }

    /// Computes the stabilizing steady-state Riccati covariance $P_\infty$ and Kalman-Bucy gain $K_\infty = P_\infty c / r$.
    pub fn steady_state_covariance_and_gain(&self) -> (f64, f64) {
        let c2_r = (self.obs_c * self.obs_c) / self.meas_r;
        let p_inf =
            (self.drift_a + (self.drift_a * self.drift_a + c2_r * self.process_q).sqrt()) / c2_r;
        let k_inf = p_inf * self.obs_c / self.meas_r;
        (p_inf, k_inf)
    }
}
