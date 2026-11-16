//! Kalman-Bucy Continuous-Time Optimal Filtering & LQG Separation Principle Riccati Equation (Kalman, 1960; Kalman & Bucy, 1961).
//!
//! ### 1. Steady-State Scalar Algebraic Riccati Equation $2 a P + q - P^2 c^2 / r = 0$
//! For $dx_t = a x_t dt + \sqrt{q} dW_t$ and observation $dy_t = c x_t dt + \sqrt{r} dV_t$, the steady-state error variance $P_\infty > 0$ and Kalman gain $K_\infty = P_\infty c / r$ satisfy:
//! $$P_\infty = \frac{r}{c^2}\!\left(a + \sqrt{a^2 + \frac{c^2 q}{r}}\right), \qquad K_\infty = \frac{a + \sqrt{a^2 + c^2 q / r}}{c}$$

use crate::error::{Error, Result};

/// Kalman-Bucy Continuous-Time Riccati Filter Engine.
#[derive(Debug, Clone, Copy)]
pub struct KalmanBucyRiccatiFilter {
    /// Drift coefficient $a \in \mathbb{R}$.
    pub drift_a: f64,
    /// Observation gain $c \neq 0$.
    pub obs_c: f64,
}

impl Default for KalmanBucyRiccatiFilter {
    fn default() -> Self {
        Self {
            drift_a: -1.0,
            obs_c: 1.0,
        }
    }
}

impl KalmanBucyRiccatiFilter {
    /// Creates a new Kalman-Bucy Riccati Filter engine.
    pub fn new(drift_a: f64, obs_c: f64) -> Result<Self> {
        if obs_c.abs() < 1e-15 || !drift_a.is_finite() {
            return Err(Error::Msg("obs_c must be non-zero".to_string()));
        }
        Ok(Self { drift_a, obs_c })
    }

    /// Evaluates the steady-state error covariance $P_\infty$ and Kalman-Bucy gain $K_\infty = P_\infty c / r$.
    pub fn steady_state_covariance_and_gain(
        &self,
        process_noise_q: f64,
        meas_noise_r: f64,
    ) -> Result<(f64, f64)> {
        if process_noise_q <= 0.0 || meas_noise_r <= 0.0 {
            return Err(Error::Msg(
                "process_noise_q and meas_noise_r must be > 0".to_string(),
            ));
        }
        let c2 = self.obs_c * self.obs_c;
        let root = (self.drift_a * self.drift_a + c2 * process_noise_q / meas_noise_r).sqrt();
        let p_inf = (meas_noise_r / c2) * (self.drift_a + root);
        let k_inf = p_inf * self.obs_c / meas_noise_r;
        Ok((p_inf, k_inf))
    }
}
