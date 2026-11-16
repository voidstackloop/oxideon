//! Pardoux-Peng Backward Stochastic Differential Equations (BSDEs) & Nonlinear Feynman-Kac Theorem (Pardoux & Peng, 1990).
//!
//! ### 1. Linear BSDE Closed-Form Solution &Recursive Risk-Neutral Discounting
//! A Backward SDE $-dY_t = f(t, Y_t, Z_t) dt - Z_t dW_t$ with terminal condition $Y_T = \xi$ and linear driver
//! $f(t, y, z) = -r y - \theta z$ has the unique adapted solution given by Girsanov's risk-neutral expectation:
//! $$Y_0 = e^{-r T} \mathbb{E}^{\mathbb{Q}}[\xi]$$

use crate::error::{Error, Result};

/// Pardoux-Peng Backward SDE (BSDE) & Nonlinear Feynman-Kac Engine.
#[derive(Debug, Clone, Copy)]
pub struct PardouxPengBsdeEngine {
    /// Discount rate $r$.
    pub rate_r: f64,
    /// Market price of risk $\theta$.
    pub risk_lambda_theta: f64,
}

impl PardouxPengBsdeEngine {
    /// Creates a new Pardoux-Peng BSDE engine.
    pub fn new(rate_r: f64, risk_lambda_theta: f64) -> Self {
        Self {
            rate_r,
            risk_lambda_theta,
        }
    }

    /// Computes a backward Euler-Maruyama conditional expectation step $Y_{t_k} = \frac{1}{1 + r \Delta t}(\mathbb{E}_k[Y_{t_{k+1}}] - \theta Z_{t_k} \Delta t)$.
    pub fn backward_bsde_step(&self, expected_y_next: f64, control_z: f64, dt: f64) -> Result<f64> {
        if dt <= 0.0 {
            return Err(Error::Msg("dt must be > 0".to_string()));
        }
        Ok((expected_y_next - self.risk_lambda_theta * control_z * dt) / (1.0 + self.rate_r * dt))
    }
}
