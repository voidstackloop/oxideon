//! Candès-Romberg-Tao Compressed Sensing, Restricted Isometry Property ($\delta_{2s} < \sqrt{2} - 1$) & Iterative Soft Thresholding (Candès, Romberg & Tao, 2006).
//!
//! ### 1. Candès $\delta_{2s} < \sqrt{2} - 1$ Exact $\ell_1$-Basis Pursuit Recovery Theorem
//! A sensing matrix $A \in \mathbb{R}^{m \times N}$ ($m \ll N$) satisfies the $2s$-Restricted Isometry Property with constant
//! $\delta_{2s} < \sqrt{2} - 1 \approx 0.41421356$, guaranteeing that $\min \|x\|_1 \text{ s.t. } A x = y$ recovers every $s$-sparse vector
//! exactly when $m \ge C \, s \ln(N / s)$.

use crate::error::{Error, Result};

/// Candès-Tao Compressed Sensing, RIP & Soft-Thresholding Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct RestrictedIsometryLassoEngine;

impl RestrictedIsometryLassoEngine {
    /// Creates a new Compressed Sensing RIP & Lasso engine.
    pub fn new() -> Self {
        Self
    }

    /// Checks Candès' sufficient Restricted Isometry Constant condition $\delta_{2s} < \sqrt{2} - 1$ and evaluates the noise stability factor $C_0(\delta_{2s})$.
    pub fn check_candes_rip_condition(delta_2s: f64) -> Result<(bool, f64)> {
        if !(0.0..1.0).contains(&delta_2s) {
            return Err(Error::Msg("delta_2s must be in [0, 1)".to_string()));
        }
        let threshold = std::f64::consts::SQRT_2 - 1.0;
        let ok = delta_2s < threshold;
        let c0 = if ok {
            2.0 * (1.0 + (1.0 - std::f64::consts::SQRT_2) * delta_2s)
                / (1.0 - (1.0 + std::f64::consts::SQRT_2) * delta_2s)
        } else {
            f64::INFINITY
        };
        Ok((ok, c0))
    }

    /// Applies the proximal $\ell_1$ Soft-Thresholding operator $\mathcal{S}_\lambda(x_i) = \operatorname{sgn}(x_i) \max(|x_i| - \lambda, 0)$.
    pub fn proximal_soft_threshold(x: &[f64], lambda: f64) -> Vec<f64> {
        x.iter()
            .map(|&v| {
                if v > lambda {
                    v - lambda
                } else if v < -lambda {
                    v + lambda
                } else {
                    0.0
                }
            })
            .collect()
    }
}
